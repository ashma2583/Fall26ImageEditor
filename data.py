import os
import re

from datasets import load_dataset
from torchvision import transforms


EDIT_CATEGORIES = (
    "Add",
    "Change Global",
    "Change Local",
    "Change Color",
    "Transform Global",
    "Transform Local",
    "Replace",
    "Turn",
    "Others",
)


def categorize_instruction(instruction):
    """Assign an approximate UltraEdit edit type from its instruction text.

    UltraEdit's public rows do not include the paper's edit-type labels, so this
    uses transparent keyword rules. Ambiguous or unmatched instructions fall
    back to ``Others``.
    """
    prompt = instruction.casefold()

    if re.search(r"\b(add|insert|include|introduce|place|put|attach)\b", prompt):
        return "Add"
    if re.search(r"\b(replace|swap|substitute|exchange)\b", prompt):
        return "Replace"

    whole_image = re.search(
        r"\b(whole|entire|overall|full)\s+(image|picture|photo|scene)\b"
        r"|\b(image|picture|photo)\s+as\s+a\s+whole\b",
        prompt,
    )
    global_scene = re.search(
        r"\b(background|scene|setting|landscape|weather|season|sky)\b"
        r"|\b(style|painting|cartoon|sketch|illustration)\s+style\b",
        prompt,
    )
    turn_into = re.search(r"\bturn\b.+\binto\b", prompt)

    if re.search(r"\b(color|colour|hue|tint|shade)\b", prompt):
        return "Change Color"
    if turn_into:
        if whole_image or global_scene:
            return "Transform Global"
        return "Turn"
    if re.search(r"\b(transform|convert)\b", prompt):
        return "Transform Global" if whole_image or global_scene else "Transform Local"
    if global_scene and re.search(r"\b(make|change|turn|transform|convert)\b", prompt):
        return "Transform Global"
    if whole_image or re.search(
        r"\b(blur|brighten|darken|sharpen|desaturate|grayscale|sepia)\b", prompt
    ):
        return "Change Global"
    if re.search(r"\b(change|modify|alter|remove|erase|adjust)\b", prompt):
        return "Change Local"
    return "Others"


def filter_edit_category(dataset, category):
    """Return a lazy dataset view containing one inferred edit category."""
    if category not in EDIT_CATEGORIES:
        raise ValueError(
            f"Unknown edit category {category!r}. Choose one of {EDIT_CATEGORIES}."
        )

    def matches(example):
        label = example.get("category")
        if label is None:
            instruction = example.get("instruction", example.get("edit_prompt", ""))
            label = categorize_instruction(instruction)
        return label == category

    return dataset.filter(matches)


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
            "category": [categorize_instruction(prompt) for prompt in batch["edit_prompt"]],
        }

    if streaming:
        dataset = dataset.map(preprocess, batched=True, batch_size=16)
    else:
        dataset.set_transform(preprocess)

    return dataset
