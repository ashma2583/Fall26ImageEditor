"""Sweep InstructPix2Pix guidance on examples saved in a training evaluation grid."""

import argparse
from pathlib import Path
from textwrap import wrap

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from inference import checkpoint_from_config, prepare_conditioning, sample_image
from models.baseline import InstructPix2PixModel
from paths import experiment_output_dir


ROOT = Path(__file__).resolve().parent


def extract_case(grid, resolution, row):
    row_height = resolution + 74
    y = 30 + row * row_height
    if grid.width != 3 * resolution or y + resolution > grid.height:
        raise ValueError(f"Evaluation grid does not contain row {row} at resolution {resolution}")
    return tuple(
        grid.crop((column * resolution, y, (column + 1) * resolution, y + resolution))
        for column in range(3)
    )


def make_grid(model, source, target, reference, instruction, row, source_scales,
              instruction_scales, steps, seed):
    resolution = source.width
    cell_height = resolution + 24
    header_height = resolution + 68
    canvas = Image.new(
        "RGB", (resolution * max(3, len(source_scales)),
                header_height + len(instruction_scales) * cell_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, (title, image) in enumerate(
        (("Source", source), ("Target", target), ("Original eval", reference))
    ):
        canvas.paste(image, (column * resolution, 0))
        draw.text((column * resolution + 4, resolution + 3), title, fill="black")
    for index, line in enumerate(wrap(f"Held-out {row}: {instruction}", width=65)[:2]):
        draw.text((4, resolution + 27 + index * 16), line, fill="black")

    conditioning = prepare_conditioning(model, source, instruction)
    baseline_error = None
    for instruction_index, instruction_scale in enumerate(instruction_scales):
        for source_index, source_scale in enumerate(source_scales):
            result = sample_image(
                model, conditioning, steps, instruction_scale, source_scale, seed,
                progress=False,
            )
            x = source_index * resolution
            y = header_height + instruction_index * cell_height
            canvas.paste(result, (x, y))
            draw.text((x + 3, y + resolution + 3),
                      f"text {instruction_scale:g}, source {source_scale:g}", fill="black")
            if steps == 30 and instruction_scale == 7.5 and source_scale == 1.5:
                baseline_error = np.abs(
                    np.asarray(result, dtype=np.int16)
                    - np.asarray(reference, dtype=np.int16)
                ).mean()
    return canvas, baseline_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.sweep_config.open(encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)

    resolution = int(config["dataset"]["resolution"])
    output_root = experiment_output_dir(config, ROOT)
    evaluation_grid = output_root / "evaluation" / sweep["evaluation_grid"]
    output_dir = output_root / "evaluation" / sweep["output_dir"]
    checkpoint = checkpoint_from_config(config)
    source_scales = [float(value) for value in sweep["source_guidance_scales"]]
    instruction_scales = [float(value) for value in sweep["instruction_guidance_scales"]]
    steps_values = [int(value) for value in sweep["steps"]]
    if not source_scales or not instruction_scales or not steps_values:
        raise ValueError("Guidance scales and steps must be nonempty")

    with Image.open(evaluation_grid) as image:
        saved_grid = image.convert("RGB")
    cases = []
    for case in sweep["cases"]:
        row = int(case["row"])
        cases.append((row, case["instruction"], *extract_case(saved_grid, resolution, row)))

    if not torch.cuda.is_available():
        raise RuntimeError("This sweep requires a CUDA GPU")
    model = InstructPix2PixModel(
        model_id=config["model"]["pretrained"],
        device="cpu",
        source_drop_rate=0.0,
        instruction_drop_rate=0.0,
        model_path=str(checkpoint),
    )
    # Training evaluation sampled in float32, so keep that precision for comparison.
    model.eval().to(device="cuda", dtype=torch.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(config["training"]["seed"]) + 10000
    print(f"Checkpoint: {checkpoint}; source grid: {evaluation_grid}", flush=True)
    for row, instruction, source, target, reference in cases:
        for steps in steps_values:
            grid, baseline_error = make_grid(
                model, source, target, reference, instruction, row,
                source_scales, instruction_scales, steps, seed_base + row,
            )
            path = output_dir / f"heldout-{row:02d}-steps-{steps:02d}.png"
            grid.save(path)
            print(f"Saved {path}; baseline mean absolute pixel difference: "
                  f"{baseline_error if baseline_error is not None else 'n/a'}", flush=True)


if __name__ == "__main__":
    main()
