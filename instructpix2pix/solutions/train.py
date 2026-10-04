"""Fine-tune SD 1.5 into InstructPix2Pix.

Sanity checks (tiny models, CPU, seconds):  python -m instructpix2pix.solutions.train --check
Overfit test on the 1k-sample set:          python -m instructpix2pix.solutions.train --max_samples 64 --max_steps 500
Paper-ish run (needs a ~24GB+ GPU):         python -m instructpix2pix.solutions.train --dataset full --streaming
"""
import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..common import SD15, Components, get_device, load_components, tiny_components, tokenize
from .data import load_split, make_collate_fn
from .model import encode_edited, encode_original, encode_text, expand_conv_in, freeze


@dataclass
class TrainConfig:
    model_id: str = SD15
    dataset: str = "small"                # key into data.DATASETS
    streaming: bool = False
    max_samples: int | None = None
    resolution: int = 256
    batch_size: int = 4
    grad_accum: int = 4                   # effective batch = 16
    lr: float = 5e-5
    max_steps: int = 15000                # optimizer steps (not micro-batches)
    conditioning_dropout: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True   # ~2x less activation memory, ~20% slower
    num_workers: int = 4
    log_every: int = 50
    save_every: int = 2000
    output_dir: str = "checkpoints/ip2p"
    seed: int = 0


def conditioning_dropout(text_emb: torch.Tensor, image_latents: torch.Tensor,
                         null_text_emb: torch.Tensor, p: float):
    """Randomly drop each condition so the model also learns the unconditional cases
    that dual classifier-free guidance needs at sampling time.

    One uniform draw r per example:
        r in [0,  p)  -> drop text only        (learns eps(z, image, ∅))
        r in [p, 2p)  -> drop both             (learns eps(z, ∅, ∅))
        r in [2p, 3p) -> drop image only       (learns eps(z, ∅, text))
    "Drop text" means use the embedding of the empty prompt "", not zeros.
    "Drop image" means zero the image latent.
    """
    r = torch.rand(text_emb.shape[0], device=text_emb.device)
    drop_text = r < 2 * p
    drop_image = (r >= p) & (r < 3 * p)
    text_emb = torch.where(drop_text[:, None, None], null_text_emb.expand_as(text_emb), text_emb)
    image_latents = image_latents * (~drop_image).to(image_latents.dtype)[:, None, None, None]
    return text_emb, image_latents


def training_step(c: Components, batch: dict, null_text_emb: torch.Tensor, p: float) -> torch.Tensor:
    """One diffusion training step. Returns the scalar MSE loss."""
    device, vae_dtype = c.unet.device, c.vae.dtype
    edited = batch["edited_pixel_values"].to(device, vae_dtype)
    original = batch["original_pixel_values"].to(device, vae_dtype)

    latents = encode_edited(c.vae, edited).float()
    noise = torch.randn_like(latents)
    t = torch.randint(0, c.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
    noisy = c.noise_scheduler.add_noise(latents, noise, t)

    text_emb = encode_text(c.text_encoder, batch["input_ids"]).float()
    image_latents = encode_original(c.vae, original).float()
    text_emb, image_latents = conditioning_dropout(text_emb, image_latents, null_text_emb.float(), p)

    pred = c.unet(torch.cat([noisy, image_latents], dim=1), t, encoder_hidden_states=text_emb).sample
    return F.mse_loss(pred.float(), noise)       # SD 1.5 is an epsilon-prediction model


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = get_device()
    amp = device.type == "cuda"
    weight_dtype = torch.bfloat16 if amp else torch.float32

    c = load_components(cfg.model_id)
    assert c.noise_scheduler.config.prediction_type == "epsilon"
    expand_conv_in(c.unet)
    freeze(c.vae).to(device, weight_dtype)
    freeze(c.text_encoder).to(device, weight_dtype)
    c.unet.to(device).train()            # trainable weights stay fp32; autocast handles speed
    if cfg.gradient_checkpointing:
        c.unet.enable_gradient_checkpointing()

    null_text_emb = encode_text(c.text_encoder, tokenize(c.tokenizer, [""]))

    ds, spec = load_split(cfg.dataset, "train", cfg.streaming, cfg.max_samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=not cfg.streaming,
        drop_last=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(c.tokenizer, spec, cfg.resolution, train=True),
    )
    opt = torch.optim.AdamW(c.unet.parameters(), lr=cfg.lr, weight_decay=1e-2)
    out = Path(cfg.output_dir)

    step, micro, running = 0, 0, 0.0
    pbar = tqdm(total=cfg.max_steps, desc="train")
    while step < cfg.max_steps:
        for batch in loader:
            with torch.autocast(device.type, dtype=weight_dtype, enabled=amp):
                loss = training_step(c, batch, null_text_emb, cfg.conditioning_dropout)
            (loss / cfg.grad_accum).backward()
            running += loss.item() / cfg.grad_accum
            micro += 1
            if micro % cfg.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(c.unet.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{running / cfg.log_every:.4f}")
                running = 0.0
            if step % cfg.save_every == 0:
                c.unet.save_pretrained(out / f"step-{step}" / "unet")
            if step >= cfg.max_steps:
                break

    c.unet.save_pretrained(out / "final" / "unet")
    print(f"saved {out / 'final' / 'unet'}  (load with pipeline.py --unet_path)")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        kind = type(f.default) if f.default is not None else int
        if kind is bool:
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=kind, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


# ---------------------------------------------------------------------------- checks

def _check():
    torch.manual_seed(0)
    B, p = 200_000, 0.1
    text = torch.ones(B, 1, 1)
    null = torch.zeros(1, 1, 1)
    img = torch.ones(B, 1, 1, 1)
    text_out, img_out = conditioning_dropout(text, img, null, p)
    t_drop = text_out.flatten() == 0
    i_drop = img_out.flatten() == 0
    frac = lambda m: m.float().mean().item()
    for name, got in [("text only", frac(t_drop & ~i_drop)), ("both", frac(t_drop & i_drop)),
                      ("image only", frac(~t_drop & i_drop)), ("neither", frac(~t_drop & ~i_drop))]:
        want = 1 - 3 * p if name == "neither" else p
        assert abs(got - want) < 0.01, f"drop {name}: expected ~{want:.2f}, got {got:.3f}"
    assert img_out.shape == img.shape and text_out.shape == text.shape
    print("✅ conditioning_dropout (each case ~p, neither ~1-3p)")

    c = tiny_components()
    expand_conv_in(c.unet)
    freeze(c.vae)
    freeze(c.text_encoder)
    c.unet.train()
    null_text_emb = encode_text(c.text_encoder, tokenize(c.tokenizer, [""]))
    batch = {
        "original_pixel_values": torch.rand(2, 3, 64, 64) * 2 - 1,
        "edited_pixel_values": torch.rand(2, 3, 64, 64) * 2 - 1,
        "input_ids": tokenize(c.tokenizer, ["make it snowy", "turn him into a cyborg"]),
    }
    loss = training_step(c, batch, null_text_emb, p=0.0)
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss should be a finite scalar, got {loss}"
    loss.backward()
    assert all(q.grad is None for q in c.vae.parameters()), "VAE must stay frozen"
    assert all(q.grad is None for q in c.text_encoder.parameters()), "text encoder must stay frozen"
    g = c.unet.conv_in.weight.grad
    assert g is not None and g[:, 4:].abs().sum() > 0, \
        "the image-latent channels of conv_in got no gradient: is the image concatenated into the UNet input?"
    print(f"✅ training_step (loss={loss.item():.3f}, gradients reach the new conv_in channels)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        main(parse_args())
