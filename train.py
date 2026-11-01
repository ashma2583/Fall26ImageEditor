"""Train the InstructPix2Pix baseline on MagicBrush."""

import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.baseline import InstructPix2PixModel
from data import load_magicbrush


ROOT = Path(__file__).resolve().parent


def collate_batch(rows):
    return {
        "source": torch.stack([torch.as_tensor(row["source"]) for row in rows]),
        "target": torch.stack([torch.as_tensor(row["target"]) for row in rows]),
        "instruction": [row["instruction"] for row in rows],
    }


def train(config):
    settings = config["training"]
    model_settings = config["model"]
    data_settings = config["dataset"]
    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(model_settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is unavailable.")

    precision = settings.get("mixed_precision", "no")
    # YAML 1.1 treats an unquoted `no` as False.
    if precision is False:
        precision = "no"
    if precision not in ("no", "fp16", "bf16"):
        raise ValueError("training.mixed_precision must be no, fp16, or bf16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use mixed_precision: no on CPU")
    if precision == "bf16" and device.type not in ("cuda", "cpu"):
        raise ValueError("bf16 training requires CUDA or CPU")

    accumulation = int(settings["gradient_accumulation_steps"])
    batch_size = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    if min(accumulation, batch_size, epochs) < 1:
        raise ValueError("gradient_accumulation_steps, batch_size, and epochs must be positive")

    train_dataset, _ = load_magicbrush(
        name=data_settings.get("name", "osunlp/MagicBrush"),
        resolution=int(data_settings["resolution"]),
        streaming=bool(data_settings.get("streaming", False)),
    )
    if data_settings.get("streaming", False):
        train_dataset = train_dataset.shuffle(
            buffer_size=int(data_settings.get("shuffle_buffer_size", 64)), seed=seed
        )
    else:
        train_dataset = train_dataset.shuffle(seed=seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)

    model = InstructPix2PixModel(
        model_id=model_settings["pretrained"],
        device=device,
        source_drop_rate=float(settings.get("source_drop_rate", 0.0)),
        instruction_drop_rate=float(settings.get("instruction_drop_rate", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        pending = 0
        total_loss = 0.0
        batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            autocast = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                        if precision != "no" else nullcontext())
            with autocast:
                loss = model(source, target, batch["instruction"])
            scaler.scale(loss / accumulation).backward()
            total_loss += loss.detach().item()
            batches += 1
            pending += 1
            if pending == accumulation:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0

        if batches == 0:
            raise RuntimeError("The training dataset yielded no batches.")
        if pending:
            # Correct the scale of a short final accumulation group.
            for parameter in model.unet.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulation / pending)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        checkpoint = output_dir / f"epoch-{epoch}" / "unet"
        model.save_model(checkpoint)
        print(f"Epoch {epoch}: mean loss {total_loss / batches:.4f}; saved {checkpoint}")

    final_checkpoint = output_dir / "final" / "unet"
    model.save_model(final_checkpoint)
    print(f"Saved final UNet to {final_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)


if __name__ == "__main__":
    main()

