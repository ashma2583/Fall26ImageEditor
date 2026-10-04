"""Model surgery and encoders: turn Stable Diffusion 1.5 into an InstructPix2Pix model.

Run the sanity checks:  python -m instructpix2pix.solutions.model
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel

from ..common import tiny_components


def expand_conv_in(unet: UNet2DConditionModel, extra_channels: int = 4) -> UNet2DConditionModel:
    """Widen the UNet's first conv so it takes [noisy latent ‖ original-image latent].

    The new input channels get zero weights, so at initialisation the modified UNet
    computes exactly what SD 1.5 did and simply ignores the image. Training then
    learns how to use it. Random init here would scramble the pretrained features.
    """
    old = unet.conv_in
    new = nn.Conv2d(
        old.in_channels + extra_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        new.bias.copy_(old.bias)
    unet.conv_in = new
    # Keep the config in sync so save_pretrained / from_pretrained round-trip correctly.
    unet.register_to_config(in_channels=new.in_channels)
    return unet


def freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


@torch.no_grad()
def encode_text(text_encoder: CLIPTextModel, input_ids: torch.Tensor) -> torch.Tensor:
    """(B, 77) token ids -> (B, 77, D) per-token embeddings for cross-attention.

    We want the full sequence (last_hidden_state), not the pooled vector.
    """
    return text_encoder(input_ids.to(text_encoder.device))[0]


@torch.no_grad()
def encode_edited(vae: AutoencoderKL, pixels: torch.Tensor) -> torch.Tensor:
    """Diffusion *target*: the edited image, in the scaled latent space SD was trained in.

    Sample from the posterior and multiply by scaling_factor (0.18215) so the latents
    have roughly unit variance, which matches the N(0, I) noise we add.
    """
    return vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def encode_original(vae: AutoencoderKL, pixels: torch.Tensor) -> torch.Tensor:
    """*Conditioning* image: the original image, concatenated onto the UNet input.

    Take the posterior mode (deterministic), and do NOT apply scaling_factor. This
    asymmetry is in the released model, so train and sample must both follow it.
    """
    return vae.encode(pixels).latent_dist.mode()


# ---------------------------------------------------------------------------- checks

def _check():
    c = tiny_components()
    unet = c.unet
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([10, 500])
    ctx = torch.randn(2, 77, unet.config.cross_attention_dim)
    with torch.no_grad():
        before = unet(x, t, encoder_hidden_states=ctx).sample

    old_weight = unet.conv_in.weight.detach().clone()
    unet = expand_conv_in(unet)
    assert unet.conv_in.in_channels == 8, "conv_in should now take 8 channels"
    assert unet.config.in_channels == 8, "remember register_to_config(in_channels=...)"
    assert torch.equal(unet.conv_in.weight[:, :4], old_weight), "first 4 channels must copy the old weights"
    assert unet.conv_in.weight[:, 4:].abs().sum() == 0, "new channels must be zero-initialised"

    with torch.no_grad():
        after = unet(torch.cat([x, torch.randn_like(x)], dim=1), t, encoder_hidden_states=ctx).sample
    assert torch.allclose(before, after, atol=1e-5), \
        "with zero-init, the expanded UNet must ignore the image and match the original output"
    print("✅ expand_conv_in")

    ids = torch.randint(0, 49408, (2, 77))
    emb = encode_text(c.text_encoder, ids)
    assert emb.shape == (2, 77, c.text_encoder.config.hidden_size), f"got {tuple(emb.shape)}"
    print("✅ encode_text")

    pixels = torch.rand(2, 3, 64, 64) * 2 - 1
    z_edit = encode_edited(c.vae, pixels)
    z_orig = encode_original(c.vae, pixels)
    assert z_edit.shape == z_orig.shape == (2, 4, 8, 8), "latents should be (B, 4, H/8, W/8)"
    assert torch.equal(z_orig, encode_original(c.vae, pixels)), "encode_original must be deterministic (.mode())"
    mode = c.vae.encode(pixels).latent_dist.mode()
    assert torch.allclose(z_orig, mode), "encode_original must NOT apply scaling_factor"
    assert not torch.allclose(z_edit, mode, atol=1e-3), "encode_edited must apply scaling_factor"
    print("✅ encode_edited / encode_original")


if __name__ == "__main__":
    _check()
