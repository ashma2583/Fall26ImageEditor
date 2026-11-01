import os
import torch
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

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def preprocess(batch):
        return {
            "source": [
                transform(img.convert("RGB"))
                for img in batch["source_img"]
            ],
            "target": [
                transform(img.convert("RGB"))
                for img in batch["target_img"]
            ],
            "instruction": batch["instruction"],
        }

    for split in dataset:
        if streaming:
            dataset[split] = dataset[split].map(preprocess, batched=True, batch_size=16)
        else:
            dataset[split].set_transform(preprocess)

    return dataset["train"], dataset["test"]
