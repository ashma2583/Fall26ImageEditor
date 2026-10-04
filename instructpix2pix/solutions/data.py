"""Dataset: (original image, edit instruction, edited image) triplets.

Run the sanity checks:  python -m instructpix2pix.solutions.data
"""
import random
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import RandomCrop

from ..common import tokenize


@dataclass
class DatasetSpec:
    repo: str
    original_col: str
    prompt_col: str
    edited_col: str


DATASETS = {
    # 1k examples (~few hundred MB): develop and overfit on this first.
    "small": DatasetSpec("fusing/instructpix2pix-1000-samples", "input_image", "edit_prompt", "edited_image"),
    # The paper's 313k CLIP-filtered set (very large, so use streaming=True).
    "full": DatasetSpec("timbrooks/instructpix2pix-clip-filtered", "original_image", "edit_prompt", "edited_image"),
}


def paired_transform(original: Image.Image, edited: Image.Image, resolution: int, train: bool = True):
    """PIL pair -> two (3, res, res) float tensors in [-1, 1], with the SAME crop and flip.

    If the two images were augmented independently, the model would be asked to learn
    a random shift/flip as part of every edit. Stacking them into one 6-channel tensor
    before augmenting guarantees they stay aligned.
    """
    x = torch.cat([TF.pil_to_tensor(original.convert("RGB")),
                   TF.pil_to_tensor(edited.convert("RGB"))])          # (6, H, W) uint8
    x = TF.resize(x, resolution, antialias=True)                      # shorter side -> res
    if train:
        i, j, h, w = RandomCrop.get_params(x, (resolution, resolution))
        x = TF.crop(x, i, j, h, w)
        if random.random() < 0.5:
            x = TF.hflip(x)
    else:
        x = TF.center_crop(x, [resolution, resolution])
    x = x.float() / 127.5 - 1.0                                       # [0, 255] -> [-1, 1]
    return x[:3], x[3:]


def load_split(name: str = "small", split: str = "train", streaming: bool = False, max_samples: int | None = None):
    spec = DATASETS[name]
    ds = load_dataset(spec.repo, split=split, streaming=streaming)
    if max_samples is not None:
        ds = ds.take(max_samples) if streaming else ds.select(range(min(max_samples, len(ds))))
    return ds, spec


def make_collate_fn(tokenizer, spec: DatasetSpec, resolution: int, train: bool = True):
    """Returns a DataLoader collate_fn that turns raw dataset rows into model-ready tensors.

    Doing the work in collate (not dataset.map) means it works the same for regular and
    streaming datasets, and augmentation is re-sampled every epoch.
    """
    def collate(rows):
        pairs = [paired_transform(r[spec.original_col], r[spec.edited_col], resolution, train) for r in rows]
        return {
            "original_pixel_values": torch.stack([p[0] for p in pairs]),
            "edited_pixel_values": torch.stack([p[1] for p in pairs]),
            "input_ids": tokenize(tokenizer, [r[spec.prompt_col] for r in rows]),
        }
    return collate


# ---------------------------------------------------------------------------- checks

def _check():
    # A gradient image is asymmetric in both axes, so any mismatched crop/flip shows up.
    w, h = 320, 240
    grad = torch.stack(torch.meshgrid(torch.linspace(0, 255, h), torch.linspace(0, 255, w), indexing="ij"))
    arr = torch.cat([grad, grad[:1].flip(1)]).permute(1, 2, 0).to(torch.uint8).numpy()
    img = Image.fromarray(arr)

    for trial in range(20):
        orig, edit = paired_transform(img, img.copy(), resolution=128, train=True)
        assert orig.shape == edit.shape == (3, 128, 128), f"got {tuple(orig.shape)}"
        assert orig.dtype == torch.float32
        assert -1.0 <= orig.min() and orig.max() <= 1.0, "pixels must be scaled to [-1, 1]"
        assert torch.equal(orig, edit), "original and edited must receive the identical crop + flip"
    print("✅ paired_transform (shape, range, shared augmentation)")

    a, _ = paired_transform(img, img, resolution=128, train=False)
    b, _ = paired_transform(img, img, resolution=128, train=False)
    assert torch.equal(a, b), "eval mode should be deterministic (center crop, no flip)"
    print("✅ paired_transform (eval is deterministic)")


if __name__ == "__main__":
    _check()
