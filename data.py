import os
import torch
from datasets import load_dataset
from torchvision import transforms

def load_magicbrush(resolution=256, streaming=False, name="osunlp/MagicBrush", include_original=False):
    """Load MagicBrush train/test splits.

    Rows yield `source` (the image this turn edits), `target` and `instruction`.
    With include_original=True they also yield `original`, the session's turn-1
    source image, for multi-turn models.
    """
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

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def to_tensor(img):
        return transform(img.convert("RGB"))

    def preprocess(batch):
        return {
            "source": [to_tensor(img) for img in batch["source_img"]],
            "target": [to_tensor(img) for img in batch["target_img"]],
            "instruction": batch["instruction"],
        }

    for split in dataset:
        if include_original and streaming:
            dataset[split] = dataset[split].map(
                _streaming_preprocess_with_original(to_tensor), batched=True, batch_size=16,
                remove_columns=["img_id", "turn_index", "source_img", "mask_img", "target_img"],
            )
        elif include_original:
            dataset[split].set_transform(_indexed_preprocess_with_original(dataset[split], to_tensor))
        elif streaming:
            dataset[split] = dataset[split].map(preprocess, batched=True, batch_size=16)
        else:
            dataset[split].set_transform(preprocess)

    return dataset["train"], dataset["test"]


def _streaming_preprocess_with_original(to_tensor):
    # MagicBrush stores each session's turns consecutively, turn 1 first, so the
    # original is the most recent turn-1 source. This relies on sequential map
    # calls, which holds for DataLoader(num_workers=0).
    state = {"img_id": None, "original": None}

    def preprocess(batch):
        out = {"source": [], "target": [], "original": [], "instruction": []}
        rows = zip(batch["img_id"], batch["turn_index"], batch["source_img"],
                   batch["target_img"], batch["instruction"])
        for img_id, turn, source_img, target_img, instruction in rows:
            source = to_tensor(source_img)
            if turn == 1:
                state["img_id"], state["original"] = img_id, source
            elif state["img_id"] != img_id:
                # A session split across shards; its turn 1 was never seen.
                continue
            out["source"].append(source)
            out["target"].append(to_tensor(target_img))
            out["original"].append(state["original"])
            out["instruction"].append(instruction)
        return out

    return preprocess


def _indexed_preprocess_with_original(split, to_tensor):
    # select_columns returns a new dataset, so it is untouched by set_transform.
    sources = split.select_columns(["source_img"])
    first_turn = {}
    for index, (img_id, turn) in enumerate(zip(split["img_id"], split["turn_index"])):
        if img_id not in first_turn or turn < first_turn[img_id][0]:
            first_turn[img_id] = (turn, index)

    def preprocess(batch):
        source = [to_tensor(img) for img in batch["source_img"]]
        original = [
            src if turn == 1 else to_tensor(sources[first_turn[img_id][1]]["source_img"])
            for src, img_id, turn in zip(source, batch["img_id"], batch["turn_index"])
        ]
        return {
            "source": source,
            "target": [to_tensor(img) for img in batch["target_img"]],
            "original": original,
            "instruction": batch["instruction"],
        }

    return preprocess
