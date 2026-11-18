import os
import re
from itertools import islice
import torch
from datasets import Image, load_dataset
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


def load_ultraedit(
    resolution=256,
    streaming=True,
    name="BleachNick/UltraEdit_500k",
    split="FreeForm",
    edit_category=None,
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
    if edit_category is not None:
        # Rejected rows do not need PIL decoding or resizing. The public release
        # has no category labels, so these labels are inferred from edit_prompt.
        dataset = dataset.cast_column("source_image", Image(decode=False))
        dataset = dataset.cast_column("edited_image", Image(decode=False))
        dataset = filter_edit_category(dataset, edit_category)
        dataset = dataset.cast_column("source_image", Image())
        dataset = dataset.cast_column("edited_image", Image())

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


def load_instructpix2pix(resolution=256, streaming=True,
                        name="timbrooks/instructpix2pix-clip-filtered"):
    """Load the original CLIP-filtered, single-turn InstructPix2Pix pairs."""
    dataset = load_dataset(
        name, split="train", streaming=streaming, token=os.getenv("HF_TOKEN")
    )
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def preprocess(batch):
        return {
            "source": [transform(image.convert("RGB")) for image in batch["original_image"]],
            "target": [transform(image.convert("RGB")) for image in batch["edited_image"]],
            "instruction": batch["edit_prompt"],
        }

    if streaming:
        return dataset.map(
            preprocess, batched=True, batch_size=16,
            remove_columns=dataset.column_names,
        )
    dataset.set_transform(preprocess)
    return dataset


def load_magicbrush_validation(resolution=256, count=24, seed=42,
                               name="osunlp/MagicBrush"):
    """Materialize a fixed, small sample from MagicBrush's dev split."""
    dataset = load_dataset(
        name,
        data_files={"validation": "data/dev-*"},
        split="validation",
        streaming=True,
        token=os.getenv("HF_TOKEN"),
    ).shuffle(buffer_size=64, seed=seed)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    examples = []
    for row in islice(dataset, count):
        examples.append({
            "source": transform(row["source_img"].convert("RGB")),
            "target": transform(row["target_img"].convert("RGB")),
            "instruction": row["instruction"],
            "img_id": row["img_id"],
        })
    if not examples:
        raise RuntimeError(f"No validation examples found in {name} dev split")
    return examples
