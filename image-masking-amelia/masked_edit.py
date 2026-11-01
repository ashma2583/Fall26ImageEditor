import argparse
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.baseline import InstructPix2PixModel  # noqa: E402

BASE = "timbrooks/instruct-pix2pix"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_magicbrush_masked(resolution=256, streaming=False, name="osunlp/MagicBrush"):
    dataset = load_dataset(
        name,
        data_files={"train": "data/train-*", "test": "data/test-*"},
        streaming=streaming,
        token=os.getenv("HF_TOKEN"),
        verification_mode="no_checks",
    )

    image_tf = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    def preprocess(batch):
        return {
            "source": [image_tf(img.convert("RGB")) for img in batch["source_img"]],
            "target": [image_tf(img.convert("RGB")) for img in batch["target_img"]],
            "mask": [(mask_tf(m.convert("L")) > 0.5).float() for m in batch["mask_img"]],
            "instruction": batch["instruction"],
        }

    for split in dataset:
        if streaming:
            dataset[split] = dataset[split].map(preprocess, batched=True, batch_size=16)
        else:
            dataset[split].set_transform(preprocess)

    return dataset["train"], dataset["test"]


def collate_batch(rows):
    return {
        "source": torch.stack([torch.as_tensor(r["source"]) for r in rows]),
        "target": torch.stack([torch.as_tensor(r["target"]) for r in rows]),
        "mask": torch.stack([torch.as_tensor(r["mask"]) for r in rows]),
        "instruction": [r["instruction"] for r in rows],
    }


class MaskedInstructPix2PixModel(InstructPix2PixModel):
    def __init__(self, model_id, device, source_drop_rate, instruction_drop_rate, mask_drop_rate=0.1):
        super().__init__(model_id, device, source_drop_rate, instruction_drop_rate)
        self.mask_drop_rate = mask_drop_rate

        old = self.unet.conv_in
        if old.in_channels == 8:
            new = nn.Conv2d(
                9, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
            ).to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :8].copy_(old.weight)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            self.unet.conv_in = new
            self.unet.register_to_config(in_channels=9)
        elif old.in_channels != 9:
            raise ValueError(f"Expected an 8- or 9-channel UNet, got {old.in_channels}")

    def forward(self, source, target, instructions, mask):
        with torch.no_grad():
            scale = self.vae.config.scaling_factor
            source_latents = self.vae.encode(source).latent_dist.sample() * scale
            target_latents = self.vae.encode(target).latent_dist.sample() * scale
            if self.training and self.source_drop_rate:
                drop = torch.rand(source.shape[0], device=source.device) < self.source_drop_rate
                source_latents = source_latents.masked_fill(drop[:, None, None, None], 0)
            if self.training and self.instruction_drop_rate:
                instructions = [
                    "" if torch.rand(()).item() < self.instruction_drop_rate else text
                    for text in instructions
                ]
            ids = self.tokenizer(
                instructions,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(self.device)
            text_encodings = self.text_encoder(ids)[0]

            m = F.interpolate(mask.to(self.device).float(), size=target_latents.shape[-2:], mode="nearest")
            if self.training and self.mask_drop_rate:
                drop_m = torch.rand(m.shape[0], device=m.device) < self.mask_drop_rate
                m = torch.where(drop_m[:, None, None, None], torch.ones_like(m), m)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],), device=self.device,
        ).long()
        noise = torch.randn_like(target_latents)
        noisy = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noisy, source_latents, m.to(noisy.dtype)], dim=1)
        pred = self.unet(model_input, timesteps, encoder_hidden_states=text_encodings).sample
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            expected = self.noise_scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            expected = noise
        return F.mse_loss(pred.float(), expected.float())


def train(args):
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    s = config["training"]
    ms = config["model"]
    ds = config["dataset"]

    seed = int(s["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device or ms["device"])
    precision = s.get("mixed_precision", "no")
    if precision is False:
        precision = "no"
    if device.type != "cuda" and precision == "fp16":
        precision = "no"

    accumulation = int(s["gradient_accumulation_steps"])
    batch_size = args.batch_size or int(s["batch_size"])
    epochs = args.epochs or int(s["epochs"])

    streaming = bool(ds.get("streaming", False))
    train_dataset, _ = load_magicbrush_masked(
        name=ds.get("name", "osunlp/MagicBrush"),
        resolution=int(ds["resolution"]),
        streaming=streaming,
    )
    if streaming:
        train_dataset = train_dataset.shuffle(buffer_size=int(ds.get("shuffle_buffer_size", 64)), seed=seed)
    else:
        train_dataset = train_dataset.shuffle(seed=seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)

    model = MaskedInstructPix2PixModel(
        model_id=ms["pretrained"],
        device=device,
        source_drop_rate=float(s.get("source_drop_rate", 0.0)),
        instruction_drop_rate=float(s.get("instruction_drop_rate", 0.0)),
        mask_drop_rate=args.mask_drop_rate,
    )
    if args.conv_in_only:
        model.unet.requires_grad_(False)
        model.unet.conv_in.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [p for p in model.unet.parameters() if p.requires_grad],
        lr=float(s["learning_rate"]),
        weight_decay=float(s["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)

    steps = 0
    done = False
    for epoch in range(1, epochs + 1):
        total, batches, pending = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                   if precision != "no" else nullcontext())
            with ctx:
                loss = model(
                    batch["source"].to(device),
                    batch["target"].to(device),
                    batch["instruction"],
                    batch["mask"].to(device),
                )
            scaler.scale(loss / accumulation).backward()
            total += loss.detach().item()
            batches += 1
            pending += 1
            if pending == accumulation:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                done = True
                break

        if pending:
            for p in model.unet.parameters():
                if p.grad is not None:
                    p.grad.mul_(accumulation / pending)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if batches == 0:
            raise RuntimeError("The training dataset yielded no batches.")
        model.save_model(out / f"epoch-{epoch}" / "unet")
        print(f"Epoch {epoch}: mean loss {total / batches:.4f}")
        if done:
            break

    model.save_model(out / "final" / "unet")
    print(f"Saved final UNet to {out / 'final' / 'unet'}")


def load_components(unet_path=None, base=BASE, device=None):
    device = device or get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(device)
    if unet_path:
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
    else:
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder.eval(),
        "vae": vae.eval(),
        "unet": unet.to(device).eval(),
        "scheduler": EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler"),
    }


def encode_text(c, text):
    tok = c["tokenizer"]
    ids = tok(
        text,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(c["unet"].device)
    return c["text_encoder"](ids)[0]


@torch.no_grad()
def masked_edit(c, image, mask, instruction, steps=30, text_guidance=7.5,
                image_guidance=1.5, seed=0, size=256, enforce=True):
    unet, vae, sched = c["unet"], c["vae"], c["scheduler"]
    device, dtype = unet.device, unet.dtype
    sf = vae.config.scaling_factor
    use_channel = unet.config.in_channels == 9

    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    mask = mask.convert("L").resize((size, size), Image.NEAREST)

    x = (torch.from_numpy(np.array(image)).float().div(127.5).sub(1)
         .permute(2, 0, 1)[None].to(device, dtype))
    m_px = torch.from_numpy((np.array(mask) > 127).astype(np.float32))[None, None].to(device, dtype)

    img_lat = vae.encode(x).latent_dist.mode()
    src = img_lat * sf
    m = F.interpolate(m_px, size=img_lat.shape[-2:], mode="nearest")

    img_cat = torch.cat([img_lat, img_lat, torch.zeros_like(img_lat)])
    m_cat = torch.cat([m, m, m])
    embeds = torch.cat([encode_text(c, instruction), encode_text(c, ""), encode_text(c, "")])

    sched.set_timesteps(steps, device=device)
    timesteps = sched.timesteps

    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(img_lat.shape, generator=gen).to(device, dtype)
    latents = noise * sched.init_noise_sigma

    for i, t in enumerate(timesteps):
        inp = sched.scale_model_input(torch.cat([latents] * 3), t)
        parts = [inp, img_cat] + ([m_cat] if use_channel else [])
        eps = unet(torch.cat(parts, dim=1), t, encoder_hidden_states=embeds).sample
        e_text, e_img, e_unc = eps.chunk(3)
        eps = e_unc + image_guidance * (e_img - e_unc) + text_guidance * (e_text - e_img)
        latents = sched.step(eps, t, latents).prev_sample

        if enforce:
            ref = sched.add_noise(src, noise, timesteps[i + 1:i + 2]) if i + 1 < len(timesteps) else src
            latents = m * latents + (1 - m) * ref

    out = vae.decode(latents / sf).sample
    out = (out / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((out * 255).astype(np.uint8))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    tr.add_argument("--output_dir", default="checkpoints/masked")
    tr.add_argument("--device", default=None)
    tr.add_argument("--batch_size", type=int, default=None)
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--mask_drop_rate", type=float, default=0.1)
    tr.add_argument("--conv_in_only", action="store_true")

    ed = sub.add_parser("edit")
    ed.add_argument("--image", default=str(here / "image.jpg"))
    ed.add_argument("--mask", default=str(here / "mask.png"))
    ed.add_argument("--instruction", default="add a white dog on the rug")
    ed.add_argument("--unet", default=None)
    ed.add_argument("--model_id", default=BASE)
    ed.add_argument("--out", default=str(here / "masked_out.png"))
    ed.add_argument("--steps", type=int, default=30)
    ed.add_argument("--size", type=int, default=256)
    ed.add_argument("--seed", type=int, default=0)
    ed.add_argument("--no_enforce", action="store_true")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        comps = load_components(args.unet, args.model_id)
        result = masked_edit(
            comps, Image.open(args.image), Image.open(args.mask), args.instruction,
            steps=args.steps, seed=args.seed, size=args.size, enforce=not args.no_enforce,
        )
        result.save(args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()