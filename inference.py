"""Edit an image with the trained InstructPix2Pix UNet.

The checkpoint is read from experiment.output_dir in the YAML config.
"""

import argparse
import os
from itertools import islice
from pathlib import Path
from textwrap import wrap

import numpy as np
import torch
import yaml
from datasets import load_dataset
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from models.baseline import InstructPix2PixModel


ROOT = Path(__file__).resolve().parent
INSTRUCTION_SCALES = (1.0, 1.2, 1.6, 2.0)
SOURCE_SCALES = (3.0, 7.5, 15.0, 25.0)
DEBUG_SAMPLES_PER_SPLIT = 5


def choose_device(configured):
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available. Use --device mps or --device cpu.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is unavailable. Use --device cpu.")
    return device


@torch.inference_mode()
def prepare_conditioning(model, source, instruction):
    """Encode the instruction and source once for all guidance combinations."""
    device = next(model.unet.parameters()).device
    dtype = next(model.unet.parameters()).dtype
    input_ids = model.tokenizer(
        [instruction, "", ""],
        padding="max_length",
        truncation=True,
        max_length=model.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    text_hidden = model.text_encoder(input_ids)[0]

    vae_dtype = next(model.vae.parameters()).dtype
    pixels = torch.from_numpy(np.asarray(source, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=vae_dtype)
    pixels = pixels / 127.5 - 1.0
    source_latents = model.vae.encode(pixels).latent_dist.mode()
    source_latents = (source_latents * model.vae.config.scaling_factor).to(dtype)
    image_conditions = torch.cat(
        [source_latents, source_latents, torch.zeros_like(source_latents)], dim=0
    )
    return text_hidden, image_conditions


@torch.inference_mode()
def sample_image(model, conditioning, steps, instruction_scale, source_scale, seed, progress=True):
    """Iteratively denoise with InstructPix2Pix dual classifier-free guidance."""
    device = next(model.unet.parameters()).device
    dtype = next(model.unet.parameters()).dtype
    text_hidden, image_conditions = conditioning
    generator = torch.Generator(device="cpu").manual_seed(seed)

    scheduler = EulerAncestralDiscreteScheduler.from_config(model.noise_scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    latents = torch.randn(image_conditions[:1].shape, generator=generator, dtype=torch.float32)
    latents = latents.to(device=device, dtype=dtype) * scheduler.init_noise_sigma

    for timestep in tqdm(scheduler.timesteps, desc="Denoising", disable=not progress):
        latent_input = torch.cat([latents] * 3, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, timestep)
        unet_input = torch.cat([latent_input, image_conditions], dim=1)
        predictions = model.unet(
            unet_input, timestep, encoder_hidden_states=text_hidden
        ).sample
        pred_text, pred_image, pred_uncond = predictions.chunk(3)
        guided_prediction = (
            pred_uncond
            + instruction_scale * (pred_text - pred_image)
            + source_scale * (pred_image - pred_uncond)
        )
        latents = scheduler.step(
            guided_prediction, timestep, latents, generator=generator
        ).prev_sample

    if dtype == torch.float16 and model.vae.config.force_upcast:
        model.vae.to(dtype=torch.float32)
        latents = latents.float()
    decoded = model.vae.decode(latents / model.vae.config.scaling_factor).sample
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    array = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def checkpoint_from_config(config):
    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    final = output_dir / "final" / "unet"
    if final.is_dir():
        return final
    epochs = []
    for path in output_dir.glob("epoch-*/unet"):
        try:
            epoch = int(path.parent.name.removeprefix("epoch-"))
        except ValueError:
            continue
        if path.is_dir():
            epochs.append((epoch, path))
    if epochs:
        return max(epochs)[1]
    raise FileNotFoundError(f"No final or epoch UNet checkpoint found in {output_dir}")


def random_debug_examples(config, seed):
    dataset = load_dataset(
        config["dataset"].get("name", "osunlp/MagicBrush"),
        data_files={"train": "data/train-*", "test": "data/test-*"},
        streaming=True,
        token=os.getenv("HF_TOKEN"),
    )
    for split_index, split in enumerate(("train", "test")):
        shuffled = dataset[split].shuffle(buffer_size=64, seed=seed + split_index)
        rows = list(islice(shuffled, DEBUG_SAMPLES_PER_SPLIT))
        if not rows:
            raise RuntimeError(f"MagicBrush {split} split yielded no examples")
        for index, row in enumerate(rows, start=1):
            yield split, index, row


def save_debug_grid(model, source, instruction, split, index, output_dir, steps, seed):
    resolution = source.width
    instruction_lines = wrap(
        f"Instruction: {instruction}",
        width=max(20, (4 * resolution - 24) // 7),
        break_long_words=True,
    )
    header_height = max(resolution + 36, 78 + 18 * len(instruction_lines))
    grid = Image.new(
        "RGB", (5 * resolution, header_height + 4 * resolution), "white"
    )
    grid.paste(source, (0, 0))
    draw = ImageDraw.Draw(grid)
    draw.text((10, resolution + 8), "Original image", fill="black")
    draw.text((resolution + 12, 10), f"{split} example {index}", fill="black")
    for line_number, line in enumerate(instruction_lines):
        draw.text((resolution + 12, 34 + line_number * 18), line, fill="black")
    for column, scale in enumerate(INSTRUCTION_SCALES):
        draw.text(((column + 1) * resolution + 10, header_height - 24),
                  f"Instruction {scale}", fill="black")

    conditioning = prepare_conditioning(model, source, instruction)
    with tqdm(total=16, desc=f"{split} grid {index}") as progress:
        for row, source_scale in enumerate(SOURCE_SCALES):
            draw.text((10, header_height + row * resolution + 10),
                      f"Source guidance {source_scale}", fill="black")
            for column, instruction_scale in enumerate(INSTRUCTION_SCALES):
                edited = sample_image(
                    model, conditioning, steps, instruction_scale, source_scale,
                    seed, progress=False,
                )
                grid.paste(edited, ((column + 1) * resolution,
                                    header_height + row * resolution))
                progress.update(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_{index:02d}.png"
    grid.save(path)
    print(f"Saved grid: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--image", type=Path, help="Source image to edit")
    parser.add_argument("--instruction", help="Edit instruction")
    parser.add_argument("--output", type=Path, default=Path("edited.png"))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), help="Override model.device")
    parser.add_argument(
        "--debug-grid", action="store_true",
        help=f"Save 4x4 grids for {DEBUG_SAMPLES_PER_SPLIT} shuffled images from each split",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    inference = config["inference"]
    device = choose_device(args.device or config["model"]["device"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    resolution = int(config["dataset"]["resolution"])
    steps = int(inference["num_inference_steps"])
    seed = int(config["training"].get("seed", 42))
    if resolution < 8 or resolution % 8:
        parser.error("resolution must be a positive multiple of 8")
    if steps < 1:
        parser.error("steps must be at least 1")

    if args.debug_grid:
        if args.image or args.instruction:
            parser.error("--image and --instruction are for single-image mode, not --debug-grid")
    elif not args.image or not args.instruction:
        parser.error("--image and --instruction are required unless --debug-grid is set")
    if args.image and not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")

    checkpoint = checkpoint_from_config(config)
    instruction_scale = float(inference.get("instruction_guidance_scale", inference.get("guidance_scale", 7.5)))
    source_scale = float(inference.get("source_guidance_scale", inference.get("image_guidance_scale", 1.5)))

    # The checkpoint stores only the UNet. Load the VAE, text encoder, tokenizer,
    # and scheduler from the same base model used during training.
    model = InstructPix2PixModel(
        model_id=config["model"]["pretrained"],
        device="cpu",
        source_drop_rate=0.0,
        instruction_drop_rate=0.0,
        model_path=str(checkpoint),
    )
    if model.unet.config.in_channels != 8:
        raise ValueError(f"Expected an 8-channel InstructPix2Pix UNet at {checkpoint}")
    model.eval().to(device=device, dtype=dtype)

    print(f"Checkpoint: {checkpoint}\nDevice: {device}")
    if args.debug_grid:
        output_dir = (args.output.parent / f"{args.output.stem}_debug"
                      if args.output.suffix else args.output)
        for split, index, example in random_debug_examples(config, seed):
            source = example["source_img"].convert("RGB").resize(
                (resolution, resolution), Image.Resampling.BICUBIC
            )
            save_debug_grid(
                model, source, example["instruction"], split, index,
                output_dir, steps, seed + index + (0 if split == "train" else 1000),
            )
    else:
        with Image.open(args.image) as image:
            source = image.convert("RGB").resize(
                (resolution, resolution), Image.Resampling.BICUBIC
            )
        conditioning = prepare_conditioning(model, source, args.instruction)
        edited = sample_image(
            model, conditioning, steps, instruction_scale, source_scale, seed
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        edited.save(args.output)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
