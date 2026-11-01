import os

from datasets import load_dataset
from torchvision import transforms


def load_magicbrush(resolution=256, streaming=False, name="osunlp/MagicBrush"):
    dataset = load_dataset(
        name,
        data_files={
            "train": "data/train-*",
            "test": "data/test-*",
        },
        streaming=streaming,
        token=os.getenv("HF_TOKEN"),
        verification_mode="no_checks",
    )

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    def preprocess(batch):
        return {
            "source": [transform(img.convert("RGB")) for img in batch["source_img"]],
            "target": [transform(img.convert("RGB")) for img in batch["target_img"]],
            "instruction": batch["instruction"],
        }

    for split in dataset:
        if streaming:
            dataset[split] = dataset[split].map(preprocess, batched=True, batch_size=16)
        else:
            dataset[split].set_transform(preprocess)

    return dataset["train"], dataset["test"]


def load_ultraedit(
    resolution=256,
    streaming=True,
    name="BleachNick/UltraEdit_500k",
    split="FreeForm",
):
    """Load UltraEdit and expose rows in the format expected by the trainer.

    UltraEdit stores its source image, edited image, and instruction as
    ``source_image``, ``edited_image``, and ``edit_prompt``. This loader maps
    those fields to ``source``, ``target``, and ``instruction`` respectively.
    """
    dataset = load_dataset(
        name,
        split=split,
        streaming=streaming,
        token=os.getenv("HF_TOKEN"),
    )

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    def preprocess(batch):
        return {
            "source": [transform(img.convert("RGB")) for img in batch["source_image"]],
            "target": [transform(img.convert("RGB")) for img in batch["edited_image"]],
            "instruction": batch["edit_prompt"],
        }

    if streaming:
        dataset = dataset.map(preprocess, batched=True, batch_size=16)
    else:
        dataset.set_transform(preprocess)

    return dataset
