"""Multi-turn evaluation on the MagicBrush test split.

Each session is edited turn by turn, with the model's own output as the next
turn's input, and every turn is scored against MagicBrush's ground truth.
"""

import argparse
import json
import math
import os
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel

from inference import checkpoint_from_config, choose_device, prepare_conditioning, sample_image
from models.baseline import InstructPix2PixModel
from models.continuous import ContinuousInstructPix2PixModel


ROOT = Path(__file__).resolve().parent
METRICS = ("l1", "clip_i", "dino", "drift_l1")
# Ground-truth pixels this close to the original count as untouched by every edit so far.
UNCHANGED_THRESHOLD = 0.05


def test_sessions(name, resolution, limit=None):
    """Yield {img_id, original, turns: [(instruction, target), ...]} per test session."""
    dataset = load_dataset(
        name,
        data_files={"test": "data/test-*"},
        streaming=True,
        token=os.getenv("HF_TOKEN"),
        verification_mode="no_checks",
    )["test"].select_columns(["img_id", "turn_index", "source_img", "target_img", "instruction"])

    def resize(img):
        return img.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)

    def sessions():
        # MagicBrush stores each session's turns consecutively, turn 1 first.
        current = None
        for row in dataset:
            if row["turn_index"] == 1:
                if current:
                    yield current
                current = {"img_id": row["img_id"], "original": resize(row["source_img"]), "turns": []}
            elif (current is None or row["img_id"] != current["img_id"]
                  or row["turn_index"] != len(current["turns"]) + 1):
                # A turn whose earlier turns were not seen.
                continue
            current["turns"].append((row["instruction"], resize(row["target_img"])))
        if current:
            yield current

    return islice(sessions(), limit)


def to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_pil(images):
    array = (images[0].float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    return Image.fromarray(array.round().clip(0, 255).astype(np.uint8))


def load_editor(config, checkpoint, device, chain, original_guidance_scale=None):
    """Return edit(original, previous, instruction, seed) -> (image, next previous)."""
    model_type = config["model"].get("type", "baseline")
    inference = config["inference"]
    steps = int(inference["num_inference_steps"])
    instruction_scale = float(inference.get("instruction_guidance_scale", inference.get("guidance_scale", 7.5)))
    image_scale = float(inference.get("source_guidance_scale", inference.get("image_guidance_scale", 1.5)))
    if original_guidance_scale is None:
        original_guidance_scale = float(inference.get("original_guidance_scale", 1.0))
    model_path = None if checkpoint == "pretrained" else str(checkpoint)

    if model_type == "continuous":
        model = ContinuousInstructPix2PixModel(
            model_id=config["model"]["pretrained"], device=device,
            previous_drop_rate=0.0, original_drop_rate=0.0, instruction_drop_rate=0.0,
            model_path=model_path,
        )
    else:
        if chain == "latent":
            raise ValueError("--chain latent needs a continuous model")
        model = InstructPix2PixModel(
            model_id=config["model"]["pretrained"], device=device,
            source_drop_rate=0.0, instruction_drop_rate=0.0, model_path=model_path,
        )
    model.eval()
    # The VAE stays in float32; the SD VAE is unstable in fp16.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.unet.to(dtype)
    model.text_encoder.to(dtype)

    settings = {
        "model_type": model_type, "checkpoint": str(checkpoint), "chain": chain, "steps": steps,
        "instruction_guidance_scale": instruction_scale, "image_guidance_scale": image_scale,
    }
    if model_type == "continuous":
        settings["original_guidance_scale"] = original_guidance_scale

        def edit(original, previous, instruction, seed):
            # A CPU generator, as in inference.sample_image, so seeds match across models.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            if not torch.is_tensor(previous):
                previous = to_tensor(previous)
            latents = model.edit(
                to_tensor(original), previous, [instruction], num_inference_steps=steps,
                guidance_scale=instruction_scale, image_guidance_scale=image_scale,
                original_guidance_scale=original_guidance_scale, generator=generator,
            )
            image = to_pil(model.decode(latents.float()))
            return image, latents if chain == "latent" else image
    else:
        def edit(original, previous, instruction, seed):
            conditioning = prepare_conditioning(model, previous, instruction)
            image = sample_image(model, conditioning, steps, instruction_scale, image_scale, seed, progress=False)
            return image, image

    return edit, settings


class Scorer:
    """MagicBrush's metrics (L1, CLIP-I, DINO) plus drift in untouched regions."""

    def __init__(self, device):
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dino = AutoModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")

    def _similarity(self, embeddings):
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return (embeddings[0] @ embeddings[1]).item()

    def score(self, prediction, target, original):
        pred = np.asarray(prediction, dtype=np.float32) / 255
        gt = np.asarray(target, dtype=np.float32) / 255
        orig = np.asarray(original, dtype=np.float32) / 255
        unchanged = np.abs(gt - orig).max(axis=-1) < UNCHANGED_THRESHOLD
        drift = np.abs(pred - orig)[unchanged].mean() if unchanged.any() else math.nan

        pair = [prediction, target]
        clip_pixels = self.clip_processor(images=pair, return_tensors="pt").pixel_values.to(self.device)
        dino_pixels = self.dino_processor(images=pair, return_tensors="pt").pixel_values.to(self.device)
        return {
            "l1": float(np.abs(pred - gt).mean()),
            # get_image_features returns a tensor in transformers 4 and an output object in 5.
            "clip_i": self._similarity(self.clip.visual_projection(
                self.clip.vision_model(pixel_values=clip_pixels).pooler_output)),
            "dino": self._similarity(self.dino(pixel_values=dino_pixels).last_hidden_state[:, 0]),
            "drift_l1": float(drift),
        }


def save_example(session, predictions, path):
    """Top row: original then ground-truth turns. Bottom row: original then model turns."""
    size = session["original"].width
    columns = 1 + len(predictions)
    grid = Image.new("RGB", (columns * size, 2 * size + 20), "white")
    draw = ImageDraw.Draw(grid)
    for row, images in enumerate(([t for _, t in session["turns"]], predictions)):
        grid.paste(session["original"], (0, 20 + row * size))
        for column, image in enumerate(images, start=1):
            grid.paste(image, (column * size, 20 + row * size))
    for column, (instruction, _) in enumerate(session["turns"], start=1):
        draw.text((column * size + 4, 4), f"{column}: {instruction}"[: size // 6], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def summarize(per_session):
    per_turn = {}
    for record in per_session:
        for turn, scores in enumerate(record["turns"], start=1):
            per_turn.setdefault(turn, []).append(scores)
    summary = {}
    for turn, rows in sorted(per_turn.items()):
        summary[turn] = {"count": len(rows)}
        for metric in METRICS:
            values = [row[metric] for row in rows if not math.isnan(row[metric])]
            summary[turn][metric] = float(np.mean(values)) if values else math.nan
    return summary


def evaluate(config, checkpoint, device, chain, limit, output_dir, save_examples, seed,
             original_guidance_scale=None):
    edit, settings = load_editor(config, checkpoint, device, chain, original_guidance_scale)
    scorer = Scorer(device)
    sessions = test_sessions(
        config["dataset"].get("name", "osunlp/MagicBrush"), int(config["dataset"]["resolution"]), limit,
    )
    per_session = []
    with torch.inference_mode():
        for index, session in enumerate(tqdm(sessions, desc="Sessions", total=limit)):
            previous, predictions, scores = session["original"], [], []
            for turn, (instruction, target) in enumerate(session["turns"], start=1):
                # The same seeds for every model, so runs are directly comparable.
                image, previous = edit(session["original"], previous, instruction, seed + 100 * index + turn)
                predictions.append(image)
                scores.append(scorer.score(image, target, session["original"]))
            per_session.append({"img_id": session["img_id"], "turns": scores})
            if index < save_examples:
                save_example(session, predictions, output_dir / "examples" / f"{index:03d}_{session['img_id']}.png")

    if not per_session:
        raise RuntimeError("The MagicBrush test split yielded no sessions")
    summary = summarize(per_session)
    results = {**settings, "sessions": len(per_session), "per_turn": summary, "per_session": per_session}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'turn':>4} {'n':>5} " + " ".join(f"{m:>9}" for m in METRICS))
    for turn, row in summary.items():
        print(f"{turn:>4} {row['count']:>5} " + " ".join(f"{row[m]:>9.4f}" for m in METRICS))
    print(f"Saved {output_dir / 'metrics.json'}")


def plot(metrics_paths, output):
    """Plot each metric against turn for several metrics.json files."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.5))
    for path in metrics_paths:
        results = json.loads(path.read_text())
        turns = sorted(results["per_turn"], key=int)
        for axis, metric in zip(axes, METRICS):
            axis.plot([int(t) for t in turns], [results["per_turn"][t][metric] for t in turns],
                      marker="o", label=path.parent.name)
    for axis, metric in zip(axes, METRICS):
        axis.set_title(metric)
        axis.set_xlabel("turn")
        axis.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument(
        "--checkpoint",
        help="UNet directory, or 'pretrained' for timbrooks/instruct-pix2pix without fine-tuning "
             "(default: the latest checkpoint in experiment.output_dir)",
    )
    parser.add_argument(
        "--chain", choices=("pixel", "latent"), default="pixel",
        help="Pass each turn to the next as an 8-bit image, or as latents (continuous model only)",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N sessions")
    parser.add_argument("--steps", type=int, help="Override inference.num_inference_steps")
    parser.add_argument("--original-guidance-scale", type=float, help="Override for the continuous model")
    parser.add_argument("--save-examples", type=int, default=8, help="Save image strips for the first N sessions")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), help="Override model.device")
    parser.add_argument("--output", type=Path, help="Results directory (default: outputs/eval/<run name>)")
    parser.add_argument("--plot", type=Path, nargs="+", metavar="METRICS_JSON",
                        help="Plot metrics.json files against each other instead of evaluating")
    args = parser.parse_args()

    if args.plot:
        plot(args.plot, args.output or ROOT / "outputs/eval/comparison.png")
        return

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.steps:
        config["inference"]["num_inference_steps"] = args.steps
    device = choose_device(args.device or config["model"]["device"])
    checkpoint = args.checkpoint or checkpoint_from_config(config)
    run_name = config["experiment"]["name"]
    if checkpoint == "pretrained":
        run_name += "-pretrained"
    if args.chain == "latent":
        run_name += "-latent"
    if args.original_guidance_scale is not None:
        run_name += f"-orig{args.original_guidance_scale:g}"
    output_dir = args.output or ROOT / "outputs/eval" / run_name

    print(f"Checkpoint: {checkpoint}\nDevice: {device}\nOutput: {output_dir}")
    evaluate(
        config, checkpoint, device, args.chain, args.limit, output_dir, args.save_examples,
        int(config["training"].get("seed", 42)), args.original_guidance_scale,
    )


if __name__ == "__main__":
    main()
