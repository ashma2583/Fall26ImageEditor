"""Compare VAE reconstruction, no-edit generation, and guidance on fixed examples."""

import argparse
import csv
from pathlib import Path
from textwrap import wrap

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from eval_guidance import extract_case, make_grid
from inference import checkpoint_from_config, prepare_conditioning, sample_image
from models.baseline import InstructPix2PixModel
from paths import experiment_output_dir


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def reconstruct_source(model, source):
    """Round-trip the source through the frozen VAE without diffusion."""
    device = next(model.vae.parameters()).device
    dtype = next(model.vae.parameters()).dtype
    pixels = torch.from_numpy(np.asarray(source, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    pixels = pixels / 127.5 - 1.0
    latents = model.vae.encode(pixels).latent_dist.mode()
    decoded = model.vae.decode(latents).sample
    array = ((decoded[0].float() / 2 + 0.5).clamp(0, 1)
             .permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def mean_absolute_pixel_error(left, right):
    return float(np.abs(np.asarray(left, dtype=np.int16)
                        - np.asarray(right, dtype=np.int16)).mean())


def make_no_edit_grid(model, source, target, reference, instruction, row,
                      source_scales, steps, seed):
    """Hold the seed fixed while changing only image guidance for an empty prompt."""
    resolution = source.width
    columns = max(4, len(source_scales))
    header_height = resolution + 68
    canvas = Image.new("RGB", (columns * resolution, header_height + resolution + 24), "white")
    draw = ImageDraw.Draw(canvas)
    vae_image = reconstruct_source(model, source)
    for column, (label, image) in enumerate((
        ("Source", source), ("Target", target),
        ("Prior eval", reference), ("VAE round-trip", vae_image),
    )):
        x = column * resolution
        canvas.paste(image, (x, 0))
        draw.text((x + 4, resolution + 3), label, fill="black")
    for line_index, line in enumerate(wrap(f"MagicBrush dev {row}: {instruction}", width=65)[:2]):
        draw.text((4, resolution + 27 + line_index * 16), line, fill="black")

    metrics = [("vae_round_trip", "", mean_absolute_pixel_error(source, vae_image))]
    conditioning = prepare_conditioning(model, source, "")
    for column, scale in enumerate(source_scales):
        image = sample_image(model, conditioning, steps, 0.0, scale, seed, progress=False)
        x = column * resolution
        canvas.paste(image, (x, header_height))
        draw.text((x + 4, header_height + resolution + 3),
                  f"no edit, source {scale:g}", fill="black")
        metrics.append(("no_edit", scale, mean_absolute_pixel_error(source, image)))
    return canvas, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostics-config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.diagnostics_config.open(encoding="utf-8") as handle:
        diagnostics = yaml.safe_load(handle)

    resolution = int(config["dataset"]["resolution"])
    output_root = experiment_output_dir(config, ROOT)
    evaluation_grid = output_root / "evaluation" / diagnostics["evaluation_grid"]
    output_dir = output_root / "evaluation" / diagnostics["output_dir"]
    checkpoint = checkpoint_from_config(config)
    if checkpoint.parent.name != "final":
        raise ValueError(f"Expected the final checkpoint, found {checkpoint}")
    source_scales = [float(value) for value in diagnostics["source_guidance_scales"]]
    instruction_scales = [float(value) for value in diagnostics["instruction_guidance_scales"]]
    no_edit_scales = [float(value) for value in diagnostics["no_edit_source_scales"]]
    steps_values = [int(value) for value in diagnostics["steps"]]
    if not all((source_scales, instruction_scales, no_edit_scales, steps_values)):
        raise ValueError("Guidance scales and step counts must be nonempty")
    if any(steps < 1 for steps in steps_values):
        raise ValueError("All denoising step counts must be positive")

    with Image.open(evaluation_grid) as image:
        saved_grid = image.convert("RGB")
    cases = []
    for case in diagnostics["cases"]:
        row = int(case["row"])
        cases.append((row, case["instruction"], *extract_case(saved_grid, resolution, row)))
    if not cases:
        raise ValueError("At least one diagnostic case is required")

    if not torch.cuda.is_available():
        raise RuntimeError("These diagnostics require a CUDA GPU")
    model = InstructPix2PixModel(
        model_id=config["model"]["pretrained"],
        device="cpu",
        source_drop_rate=0.0,
        instruction_drop_rate=0.0,
        model_path=str(checkpoint),
    )
    model.eval().to(device="cuda", dtype=torch.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(config["training"]["seed"]) + 10000
    print(f"Final checkpoint: {checkpoint}; source grid: {evaluation_grid}", flush=True)

    metrics_path = output_dir / "no-edit-metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row", "denoising_steps", "method", "source_guidance", "mae_from_source"))
        for row, instruction, source, target, reference in cases:
            for steps in steps_values:
                seed = seed_base + row
                no_edit_grid, metrics = make_no_edit_grid(
                    model, source, target, reference, instruction, row,
                    no_edit_scales, steps, seed,
                )
                no_edit_path = output_dir / f"magicbrush-dev-{row:02d}-no-edit-{steps:02d}.png"
                no_edit_grid.save(no_edit_path)
                for method, scale, error in metrics:
                    writer.writerow((row, steps, method, scale, f"{error:.4f}"))
                print(f"Saved {no_edit_path}", flush=True)

                guidance_grid, _ = make_grid(
                    model, source, target, reference, instruction, row,
                    source_scales, instruction_scales, steps, seed,
                    case_label="MagicBrush dev", reference_label="Prior eval",
                )
                guidance_path = output_dir / f"magicbrush-dev-{row:02d}-guidance-{steps:02d}.png"
                guidance_grid.save(guidance_path)
                print(f"Saved {guidance_path}", flush=True)
    print(f"Saved {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
