"""Inference: edit an image with an instruction using dual classifier-free guidance.

Sanity checks (tiny models):     python -m instructpix2pix.solutions.pipeline --check
Authors' pretrained weights:     python -m instructpix2pix.solutions.pipeline --image in.jpg --prompt "make it snowy"
Your trained UNet:               ... --unet_path checkpoints/ip2p/final/unet
"""
import argparse
import sys

import torch
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
import torchvision.transforms.functional as TF

from ..common import IP2P, Components, get_device, load_components, tiny_components, tokenize
from .model import encode_original, encode_text, expand_conv_in


def combine_guidance(eps: torch.Tensor, guidance_scale: float, image_guidance_scale: float) -> torch.Tensor:
    """Dual classifier-free guidance (paper Eq. 3).

    eps is the UNet output for the stacked batch [text+image, image-only, unconditional].

        eps = eps_uncond
            + s_I * (eps_img  - eps_uncond)   # push toward "consistent with the input image"
            + s_T * (eps_text - eps_img)      # push toward "follows the instruction"
    """
    eps_text, eps_img, eps_uncond = eps.chunk(3)
    return (eps_uncond
            + image_guidance_scale * (eps_img - eps_uncond)
            + guidance_scale * (eps_text - eps_img))


class InstructPix2PixPipeline:
    def __init__(self, c: Components, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.device = device or get_device()
        self.dtype = dtype
        self.vae = c.vae.to(self.device, dtype).eval()
        self.text_encoder = c.text_encoder.to(self.device, dtype).eval()
        self.unet = c.unet.to(self.device, dtype).eval()
        self.tokenizer = c.tokenizer
        # Same beta schedule the model was trained with, but a better sampler than DDPM.
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(c.noise_scheduler.config)

    @classmethod
    def from_pretrained(cls, model_id: str = IP2P, unet_path: str | None = None, **kw):
        return cls(load_components(model_id, unet_path), **kw)

    def encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        """-> (3B, 77, D) stacked as [text, null, null] to line up with combine_guidance."""
        cond = encode_text(self.text_encoder, tokenize(self.tokenizer, prompts))
        null = encode_text(self.text_encoder, tokenize(self.tokenizer, [""] * len(prompts)))
        return torch.cat([cond, null, null])

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> (3B, 4, H/8, W/8) stacked as [img, img, zeros]."""
        z = encode_original(self.vae, pixels.to(self.device, self.dtype))
        return torch.cat([z, z, torch.zeros_like(z)])

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL -> (B, 3, H, W) in [-1, 1], sides rounded down to a multiple of 8 for the VAE."""
        w, h = images[0].size
        w, h = w - w % 8, h - h % 8
        x = torch.stack([TF.to_tensor(im.convert("RGB").resize((w, h), Image.LANCZOS)) for im in images])
        return x * 2 - 1

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> list[Image.Image]:
        x = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        x = (x.float() / 2 + 0.5).clamp(0, 1)
        return [TF.to_pil_image(im) for im in x.cpu()]

    @torch.no_grad()
    def __call__(self, image: Image.Image | list[Image.Image], prompt: str | list[str],
                 num_inference_steps: int = 20, guidance_scale: float = 7.5,
                 image_guidance_scale: float = 1.5, seed: int | None = None) -> list[Image.Image]:
        images = [image] if isinstance(image, Image.Image) else image
        prompts = [prompt] * len(images) if isinstance(prompt, str) else prompt
        generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

        prompt_embeds = self.encode_prompt(prompts)                     # (3B, 77, D)
        image_latents = self.encode_image(self.preprocess(images))      # (3B, 4, h, w)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        shape = (len(images), 4, *image_latents.shape[-2:])
        latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            x = self.scheduler.scale_model_input(torch.cat([latents] * 3), t)
            x = torch.cat([x, image_latents.float()], dim=1).to(self.dtype)      # (3B, 8, h, w)
            eps = self.unet(x, t, encoder_hidden_states=prompt_embeds).sample.float()
            eps = combine_guidance(eps, guidance_scale, image_guidance_scale)
            latents = self.scheduler.step(eps, t, latents, generator=generator).prev_sample

        return self.decode(latents.to(self.dtype))


# ---------------------------------------------------------------------------- checks

def _check():
    text, img, unc = torch.full((1, 4, 2, 2), 3.0), torch.full((1, 4, 2, 2), 2.0), torch.full((1, 4, 2, 2), 1.0)
    eps = torch.cat([text, img, unc])
    assert torch.allclose(combine_guidance(eps, 1.0, 1.0), text), "s_T = s_I = 1 should reduce to eps_text"
    assert torch.allclose(combine_guidance(eps, 0.0, 0.0), unc), "s_T = s_I = 0 should reduce to eps_uncond"
    assert torch.allclose(combine_guidance(eps, 0.0, 1.0), img), "s_T = 0, s_I = 1 should reduce to eps_img"
    assert torch.allclose(combine_guidance(eps, 7.5, 1.5), torch.full_like(text, 1 + 1.5 * 1 + 7.5 * 1))
    print("✅ combine_guidance")

    c = tiny_components()
    expand_conv_in(c.unet)
    pipe = InstructPix2PixPipeline(c, device=torch.device("cpu"))

    pe = pipe.encode_prompt(["make it snowy", "add fireworks"])
    assert pe.shape == (6, 77, 32), f"encode_prompt: expected (3B, 77, D), got {tuple(pe.shape)}"
    assert torch.equal(pe[2], pe[4]) and torch.equal(pe[4], pe[5]), "rows B: must all be the empty-prompt embedding"
    assert not torch.equal(pe[0], pe[2]), "rows :B must be the real prompts"
    print("✅ encode_prompt  [text, null, null]")

    il = pipe.encode_image(torch.rand(2, 3, 64, 64) * 2 - 1)
    assert il.shape == (6, 4, 8, 8), f"encode_image: expected (3B, 4, H/8, W/8), got {tuple(il.shape)}"
    assert torch.equal(il[:2], il[2:4]) and il[4:].abs().sum() == 0, "expected [img, img, zeros]"
    print("✅ encode_image   [img, img, zeros]")

    src = Image.new("RGB", (100, 75), "orange")
    out = pipe(src, "make it snowy", num_inference_steps=3, seed=0)
    assert len(out) == 1 and out[0].size == (96, 72), f"expected one 96x72 image, got {[o.size for o in out]}"
    again = pipe(src, "make it snowy", num_inference_steps=3, seed=0)
    assert torch.equal(TF.pil_to_tensor(out[0]), TF.pil_to_tensor(again[0])), "same seed should give the same image"
    print("✅ full sampling loop (shapes, determinism)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="edited.png")
    p.add_argument("--model_id", default=IP2P)
    p.add_argument("--unet_path", default=None)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)
    p.add_argument("--resolution", type=int, default=512, help="longer side is resized to this")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = InstructPix2PixPipeline.from_pretrained(a.model_id, a.unet_path, device=device, dtype=dtype)
    image = Image.open(a.image).convert("RGB")
    image.thumbnail((a.resolution, a.resolution))
    out = pipe(image, a.prompt, a.steps, a.guidance_scale, a.image_guidance_scale, a.seed)[0]
    out.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        main()
