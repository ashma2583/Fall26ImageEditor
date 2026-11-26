"""Compare saved epoch-8 outputs with the released InstructPix2Pix model."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from textwrap import wrap

import torch
import yaml
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
from PIL import Image, ImageDraw

from paths import experiment_output_dir


ROOT = Path(__file__).resolve().parent
LABEL_HEIGHT = 32


def read_manifest(sweep_dir):
    path = sweep_dir / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty sweep manifest: {path}")
    cases = defaultdict(dict)
    for row in rows:
        key = (row["case"], int(row["resolution"]))
        setting = (int(row["steps"]), float(row["text_guidance"]),
                   float(row["source_guidance"]))
        image_path = sweep_dir / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing saved model image: {image_path}")
        if setting in cases[key]:
            raise ValueError(f"Duplicate setting {key} {setting}")
        cases[key][setting] = row
    return cases


def draw_cell(canvas, draw, image, col, row, resolution, label):
    x = col * resolution
    y = row * (resolution + LABEL_HEIGHT)
    if image is not None:
        canvas.paste(image, (x, y))
    draw.text((x + 6, y + resolution + 5), label, fill="black")


def save_comparison_grid(path, resolution, headers, rows, instruction):
    width = 3 * resolution
    height = (1 + len(rows)) * (resolution + LABEL_HEIGHT) + 62
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (image, label) in enumerate(headers):
        draw_cell(canvas, draw, image, column, 0, resolution, label)
    for row_index, row in enumerate(rows, 1):
        for column, (image, label) in enumerate(row):
            draw_cell(canvas, draw, image, column, row_index, resolution, label)
    for line_index, line in enumerate(wrap(instruction, width=100)[:3]):
        draw.text((6, (1 + len(rows)) * (resolution + LABEL_HEIGHT) + 5 + 17 * line_index),
                  line, fill="black")
    canvas.save(path)
    print(f"Saved comparison grid {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sweep = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    sweep_dir = experiment_output_dir(config, ROOT) / "evaluation" / sweep["output_dir"]
    cases = read_manifest(sweep_dir)
    resolutions = [int(value) for value in sweep["resolutions"]]
    steps = [int(value) for value in sweep["step_counts"]]
    guidance_steps = int(sweep["guidance_steps"])
    text_scales = [float(value) for value in sweep["instruction_guidance_scales"]]
    source_scales = [float(value) for value in sweep["source_guidance_scales"]]
    default_text = float(config["inference"]["instruction_guidance_scale"])
    default_source = float(config["inference"]["source_guidance_scale"])
    expected = {(step, default_text, default_source) for step in steps}
    expected.update((guidance_steps, text, source)
                    for text in text_scales for source in source_scales)
    expected_cases = {f"heldout-{index:02d}" for index in range(int(sweep["heldout_examples"]))}
    if sweep.get("include_text_probe", False):
        expected_cases.add("text-probe")
    if set(cases) != {(name, res) for name in expected_cases for res in resolutions}:
        raise ValueError("Saved sweep cases do not match the comparison configuration")
    for key, records in cases.items():
        if set(records) != expected:
            raise ValueError(f"Saved sweep has missing or extra settings for {key}")
        instructions = {row["instruction"] for row in records.values()}
        seeds = {int(row["seed"]) for row in records.values()}
        if len(instructions) != 1 or len(seeds) != 1:
            raise ValueError(f"Instruction or seed mismatch within saved sweep: {key}")
        if not (sweep_dir / f"{key[0]}-{key[1]}px-steps.png").is_file():
            raise FileNotFoundError(f"Missing source grid for {key}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the comparison")

    model_id = sweep["reference_model"]
    print(f"Loading released InstructPix2Pix model: {model_id}", flush=True)
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    output_dir = sweep_dir / sweep["comparison_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "comparison-manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "instruction", "resolution", "steps", "text_guidance",
                         "source_guidance", "seed", "our_image", "ip2p_image"))
        for (name, resolution), records in sorted(cases.items()):
            with Image.open(sweep_dir / f"{name}-{resolution}px-steps.png") as image:
                source = image.crop((0, 0, resolution, resolution)).convert("RGB")
                target = image.crop((resolution, 0, 2 * resolution, resolution)).convert("RGB")
            instruction = next(iter(records.values()))["instruction"]
            results = {}
            for setting, record in sorted(records.items()):
                step_count, text_scale, source_scale = setting
                seed = int(record["seed"])
                generator = torch.Generator(device="cpu").manual_seed(seed)
                reference = pipeline(
                    prompt=instruction, image=source,
                    num_inference_steps=step_count,
                    guidance_scale=text_scale,
                    image_guidance_scale=source_scale,
                    generator=generator,
                ).images[0].convert("RGB")
                if reference.size != (resolution, resolution):
                    raise ValueError(f"Reference output has wrong size: {reference.size}")
                filename = (f"{name}-{resolution}px-{step_count}steps-"
                            f"text{text_scale:g}-source{source_scale:g}-ip2p.png")
                reference.save(output_dir / filename)
                with Image.open(sweep_dir / record["image"]) as image:
                    ours = image.convert("RGB")
                results[setting] = (ours, reference)
                writer.writerow((name, instruction, resolution, step_count, text_scale,
                                 source_scale, seed, record["image"], filename))
                handle.flush()
                print(f"Saved {filename}", flush=True)

            step_rows = [
                [(results[(step, default_text, default_source)][model_index],
                  f"{'Ours' if model_index == 0 else 'Released IP2P'}: {step} steps")
                 for step in steps]
                for model_index in (0, 1)
            ]
            headers = [(source, "Source"), (target if name != "text-probe" else None, "Target"),
                       (None, f"{name}; {resolution} px")]
            save_comparison_grid(output_dir / f"{name}-{resolution}px-steps-comparison.png",
                                 resolution, headers, step_rows, instruction)

            guidance_rows = []
            for text_scale in text_scales:
                for model_index in (0, 1):
                    guidance_rows.append([
                        (results[(guidance_steps, text_scale, source_scale)][model_index],
                         f"{'Ours' if model_index == 0 else 'Released IP2P'}: "
                         f"text {text_scale:g}, source {source_scale:g}")
                        for source_scale in source_scales
                    ])
            save_comparison_grid(output_dir / f"{name}-{resolution}px-guidance-comparison.png",
                                 resolution, headers, guidance_rows, instruction)
    print(f"Completed matched comparison in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
