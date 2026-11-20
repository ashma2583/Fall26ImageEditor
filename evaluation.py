"""Fixed image and denoising-loss checks for baseline training jobs."""

import csv
from contextlib import nullcontext
from pathlib import Path
from textwrap import wrap

import torch
from PIL import Image, ImageDraw

from inference import prepare_conditioning, sample_image


def _to_image(tensor):
    pixels = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(pixels.permute(1, 2, 0).round().byte().numpy())


def evaluate_baseline(model, examples, output_dir, config, epoch, step, precision, split=None):
    """Log repeatable denoising loss and a source/target/generated comparison."""
    if isinstance(examples, dict):
        return {
            name: evaluate_baseline(model, rows, output_dir, config, epoch, step, precision, split=name)
            for name, rows in examples.items()
        }

    device = next(model.unet.parameters()).device
    eval_settings = config["evaluation"]
    resolution = int(config["dataset"]["resolution"])
    batch_size = int(eval_settings.get("batch_size", 4))
    num_images = int(eval_settings.get("num_images", 3))
    steps = int(eval_settings.get("num_inference_steps", 20))
    seed = int(config["training"].get("seed", 42)) + 10000
    guidance = float(config["inference"].get("instruction_guidance_scale", 7.5))
    source_guidance = float(config["inference"].get("source_guidance_scale", 1.5))
    eval_dir = Path(output_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            total_loss = 0.0
            total_examples = 0
            for start in range(0, len(examples), batch_size):
                rows = examples[start:start + batch_size]
                source = torch.stack([row["source"] for row in rows]).to(device)
                target = torch.stack([row["target"] for row in rows]).to(device)
                instructions = [row["instruction"] for row in rows]
                autocast = (torch.autocast(device_type=device.type,
                                           dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
                            if precision != "no" else nullcontext())
                with autocast:
                    loss = model(source, target, instructions)
                total_loss += loss.item() * len(rows)
                total_examples += len(rows)
            validation_loss = total_loss / total_examples

            chosen = []
            seen_ids = set()
            for row in examples:
                if row["img_id"] not in seen_ids:
                    chosen.append(row)
                    seen_ids.add(row["img_id"])
                if len(chosen) == num_images:
                    break
            row_height = resolution + 74
            canvas = Image.new("RGB", (3 * resolution, 30 + row_height * len(chosen)), "white")
            draw = ImageDraw.Draw(canvas)
            for col, title in enumerate(("Source", "Target", "Generated")):
                draw.text((col * resolution + 8, 8), title, fill="black")
            for index, row in enumerate(chosen):
                source_image = _to_image(row["source"])
                target_image = _to_image(row["target"])
                conditioning = prepare_conditioning(model, source_image, row["instruction"])
                generated = sample_image(
                    model, conditioning, steps, guidance, source_guidance,
                    seed + index, progress=False,
                )
                y = 30 + index * row_height
                for col, image in enumerate((source_image, target_image, generated)):
                    canvas.paste(image, (col * resolution, y))
                instruction_lines = wrap(f"{row['img_id']}: {row['instruction']}", width=100)
                for line_index, line in enumerate(instruction_lines[:3]):
                    draw.text((8, y + resolution + 5 + line_index * 18), line, fill="black")

            image_name = f"epoch-{epoch:02d}-step-{step:06d}.png"
            if split is not None:
                image_name = f"{split}-{image_name}"
            image_path = eval_dir / image_name
            canvas.save(image_path)
            metrics_path = eval_dir / ("metrics-by-split.csv" if split is not None else "metrics.csv")
            needs_header = not metrics_path.exists()
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if needs_header:
                    if split is None:
                        writer.writerow(("epoch", "optimizer_step", "validation_mse", "examples", "image"))
                    else:
                        writer.writerow(("split", "epoch", "optimizer_step", "denoising_mse", "examples", "image"))
                values = (epoch, step, f"{validation_loss:.6f}", total_examples, image_path.name)
                writer.writerow((split, *values) if split is not None else values)
            label = f" {split}" if split is not None else ""
            print(f"Evaluation{label} epoch {epoch} step {step}: MSE {validation_loss:.5f}; "
                  f"images {image_path}", flush=True)
            return validation_loss
    finally:
        if was_training:
            model.train()
            model.vae.eval()
            model.text_encoder.eval()
