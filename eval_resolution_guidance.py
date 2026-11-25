"""Compare held-out edits across output resolutions, step counts, and guidance."""

import argparse
import csv
from itertools import islice
from pathlib import Path
from textwrap import wrap

import numpy as np
import torch
import yaml
from datasets import Image as DatasetImage
from PIL import Image, ImageDraw, ImageFont

from data import load_ultraedit
from eval_guidance import extract_case
from inference import prepare_conditioning, sample_image
from models.baseline import InstructPix2PixModel
from paths import experiment_output_dir, has_saved_unet


ROOT = Path(__file__).resolve().parent


def tensor_image(tensor):
    pixels = ((torch.as_tensor(tensor).float().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(pixels.permute(1, 2, 0).round().byte().numpy())


def decode_image(value):
    return (DatasetImage().decode_example(value) if isinstance(value, dict) else value).convert("RGB")


def vae_roundtrip(model, conditioning):
    source_latents = conditioning[1][:1]
    with torch.inference_mode():
        decoded = model.vae.decode(source_latents).sample[0]
    pixels = ((decoded.float().cpu().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(pixels.permute(1, 2, 0).round().byte().numpy())


def draw_panel(canvas, draw, image, col, row, resolution, label, label_height):
    x = col * resolution
    y = row * (resolution + label_height)
    if image is not None:
        canvas.paste(image, (x, y))
    draw.text((x + 6, y + resolution + 5), label, fill="black")


def save_grid(path, source, target, vae_source, reference, results, instruction,
              resolution, columns, label_height=32):
    rows = 1 + (len(results) + columns - 1) // columns
    footer = 60
    canvas = Image.new("RGB", (columns * resolution,
                               rows * (resolution + label_height) + footer), "white")
    draw = ImageDraw.Draw(canvas)
    header = [(source, "Source"), (target, "Target"),
              (vae_source, "VAE source roundtrip"), (reference, "Saved epoch 8 edit")]
    for col, (image, label) in enumerate(header[:columns]):
        draw_panel(canvas, draw, image, col, 0, resolution, label, label_height)
    for index, (image, label) in enumerate(results):
        row, col = divmod(index, columns)
        draw_panel(canvas, draw, image, col, row + 1, resolution, label, label_height)
    for index, line in enumerate(wrap(instruction, width=100)[:3]):
        draw.text((6, rows * (resolution + label_height) + 4 + 17 * index),
                  line, fill="black")
    canvas.save(path)


def text_probe():
    """A simple, clearly legible text-edit stress test with no target image."""
    image = Image.new("RGB", (512, 512), "#d2c8b3")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 125, 442, 390), fill="#173c39", outline="#ebdfb2", width=14)
    try:
        font = ImageFont.load_default(size=108)
        draw.text((256, 256), "OPEN", font=font, fill="white", anchor="mm")
    except TypeError:
        # Older Pillow versions do not accept a font size for the default font.
        font = ImageFont.load_default()
        bounds = draw.textbbox((0, 0), "OPEN", font=font, stroke_width=1)
        mask = Image.new("L", (bounds[2] - bounds[0], bounds[3] - bounds[1]))
        ImageDraw.Draw(mask).text((-bounds[0], -bounds[1]), "OPEN", font=font,
                                  fill=255, stroke_width=1, stroke_fill=255)
        image.paste("white", (106, 211), mask.resize((300, 90), Image.Resampling.NEAREST))
    return {"name": "text-probe", "instruction": "Change the word OPEN on the sign to CLOSED.",
            "source_native": image, "target_native": None, "source_256": None,
            "target_256": None, "reference": None, "seed_offset": 100}


def heldout_cases(config, eval_config, reference_grid):
    data_config = config["dataset"]
    dataset = load_ultraedit(
        name=data_config["name"], split=data_config.get("split", "FreeForm"),
        resolution=int(data_config["resolution"]), streaming=True,
        min_scores=data_config.get("min_scores"),
    ).shuffle(buffer_size=int(data_config.get("shuffle_buffer_size", 1000)),
              seed=int(config["training"]["seed"]))
    count = int(eval_config["heldout_examples"])
    rows = list(islice(dataset, count))
    if len(rows) != count:
        raise RuntimeError(f"Only {len(rows)} held-out rows found; expected {count}")
    cases = []
    for index, row in enumerate(rows):
        source_256 = tensor_image(row["source"])
        target_256 = tensor_image(row["target"])
        saved_source, saved_target, reference = extract_case(reference_grid, 256, index)
        source_error = np.abs(np.asarray(source_256, dtype=np.int16)
                              - np.asarray(saved_source, dtype=np.int16)).mean()
        target_error = np.abs(np.asarray(target_256, dtype=np.int16)
                              - np.asarray(saved_target, dtype=np.int16)).mean()
        if max(source_error, target_error) > 1:
            raise RuntimeError(f"Held-out row {index} differs from saved epoch-8 grid: "
                               f"source MAE {source_error:.2f}, target MAE {target_error:.2f}")
        cases.append({"name": f"heldout-{index:02d}", "instruction": row["instruction"],
                      "source_native": decode_image(row["source_image"]),
                      "target_native": decode_image(row["edited_image"]),
                      "source_256": source_256, "target_256": target_256,
                      "reference": reference, "seed_offset": index})
        print(f"Verified held-out row {index}: {row['instruction']}", flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sweep = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    if int(config["dataset"]["resolution"]) != 256:
        raise ValueError("This evaluation expects the 256-pixel training config")
    resolutions = [int(value) for value in sweep["resolutions"]]
    step_counts = [int(value) for value in sweep["step_counts"]]
    text_scales = [float(value) for value in sweep["instruction_guidance_scales"]]
    source_scales = [float(value) for value in sweep["source_guidance_scales"]]
    guidance_steps = int(sweep["guidance_steps"])
    if any(res < 8 or res % 8 for res in resolutions) or any(step < 1 for step in step_counts):
        raise ValueError("Resolutions must be multiples of 8 and steps must be positive")
    if not resolutions or not step_counts or not text_scales or not source_scales:
        raise ValueError("Evaluation sweep settings must be nonempty")
    default_text = float(config["inference"]["instruction_guidance_scale"])
    default_source = float(config["inference"]["source_guidance_scale"])
    if guidance_steps not in step_counts or default_text not in text_scales or default_source not in source_scales:
        raise ValueError("The guidance grid must include the default setting and step count")
    output_root = experiment_output_dir(config, ROOT)
    checkpoint = output_root / f"epoch-{int(sweep['checkpoint_epoch'])}" / "unet"
    if not has_saved_unet(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    reference_path = output_root / "evaluation" / sweep["reference_grid"]
    with Image.open(reference_path) as image:
        reference_grid = image.convert("RGB")
    cases = heldout_cases(config, sweep, reference_grid)
    if sweep.get("include_text_probe", False):
        cases.append(text_probe())
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this evaluation")
    model = InstructPix2PixModel(
        model_id=config["model"]["pretrained"], device="cpu", source_drop_rate=0.0,
        instruction_drop_rate=0.0, model_path=str(checkpoint),
    )
    model.eval().to(device="cuda", dtype=torch.float32)
    output_dir = output_root / "evaluation" / sweep["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(config["training"]["seed"]) + 10000
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "instruction", "resolution", "steps", "text_guidance",
                         "source_guidance", "seed", "image"))
        for case in cases:
            for resolution in resolutions:
                source = (case["source_256"] if resolution == 256 and case["source_256"] is not None
                          else case["source_native"].resize((resolution, resolution), Image.Resampling.BILINEAR))
                target = (case["target_256"] if resolution == 256 and case["target_256"] is not None
                          else (case["target_native"].resize((resolution, resolution), Image.Resampling.BILINEAR)
                                if case["target_native"] is not None else None))
                reference = case["reference"] if resolution == 256 else None
                conditioning = prepare_conditioning(model, source, case["instruction"])
                vae_source = vae_roundtrip(model, conditioning)
                seed = seed_base + case["seed_offset"]
                generated = {}

                def sample(steps, text_scale, source_scale):
                    key = (steps, text_scale, source_scale)
                    if key not in generated:
                        image = sample_image(model, conditioning, steps, text_scale,
                                             source_scale, seed, progress=False)
                        path = output_dir / (f"{case['name']}-{resolution}px-{steps}steps-"
                                             f"text{text_scale:g}-source{source_scale:g}.png")
                        image.save(path)
                        writer.writerow((case["name"], case["instruction"], resolution, steps,
                                         text_scale, source_scale, seed, path.name))
                        handle.flush()
                        generated[key] = image
                        print(f"Saved {path.name}", flush=True)
                    return generated[key]

                step_results = [(sample(steps, default_text, default_source), f"{steps} steps")
                                for steps in step_counts]
                save_grid(output_dir / f"{case['name']}-{resolution}px-steps.png",
                          source, target, vae_source, reference, step_results,
                          case["instruction"], resolution, columns=4)
                guidance_results = [
                    (sample(guidance_steps, text_scale, source_scale),
                     f"text {text_scale:g}, source {source_scale:g}")
                    for text_scale in text_scales for source_scale in source_scales
                ]
                save_grid(output_dir / f"{case['name']}-{resolution}px-guidance.png",
                          source, target, vae_source, reference, guidance_results,
                          case["instruction"], resolution, columns=3)
                if reference is not None:
                    baseline = sample(step_counts[0], default_text, default_source)
                    error = np.abs(np.asarray(baseline, dtype=np.int16)
                                   - np.asarray(reference, dtype=np.int16)).mean()
                    print(f"{case['name']}: saved-eval reproduction MAE {error:.2f}/255", flush=True)
    print(f"Completed {len(cases)} cases at {len(resolutions)} resolutions: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
