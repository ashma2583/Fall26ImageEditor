"""Shared plumbing: loading pretrained components, tokenization, tiny test models.

Nothing in this file is a TODO. The interesting code lives in model.py, data.py,
train.py and pipeline.py.
"""
from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

# The original runwayml/stable-diffusion-v1-5 repo was taken down; this is the official mirror.
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# Authors' released InstructPix2Pix weights (UNet already has 8 input channels).
IP2P = "timbrooks/instruct-pix2pix"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Components:
    vae: AutoencoderKL                  # image (3, H, W) in [-1, 1]  <->  latent (4, H/8, W/8)
    text_encoder: CLIPTextModel         # token ids (77,)  ->  embeddings (77, 768)
    tokenizer: CLIPTokenizer
    unet: UNet2DConditionModel          # the only thing we train
    noise_scheduler: DDPMScheduler      # beta schedule + add_noise() for training


def load_components(model_id: str = SD15, unet_path: str | None = None) -> Components:
    """Load the five Stable Diffusion pieces from a Hugging Face repo.

    unet_path: optionally swap in a UNet you trained (a folder saved by unet.save_pretrained).
    """
    unet = (UNet2DConditionModel.from_pretrained(unet_path) if unet_path
            else UNet2DConditionModel.from_pretrained(model_id, subfolder="unet"))
    return Components(
        vae=AutoencoderKL.from_pretrained(model_id, subfolder="vae"),
        text_encoder=CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
        unet=unet,
        noise_scheduler=DDPMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )


def tokenize(tokenizer: CLIPTokenizer, prompts: list[str]) -> torch.Tensor:
    """List of strings -> (B, 77) int64 token ids, padded/truncated to CLIP's context length."""
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def tiny_components() -> Components:
    """Randomly initialised, architecturally faithful miniature of SD 1.5.

    Same shapes where it matters (4 latent channels, 8x VAE downsampling, 77-token CLIP
    context, 4-channel UNet input before surgery) but ~1000x fewer parameters, so every
    sanity check runs on a laptop CPU in seconds. Outputs are noise; only shapes and
    wiring are being tested.
    """
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        latent_channels=4,
    )
    text_encoder = CLIPTextModel(CLIPTextConfig(
        vocab_size=49408,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=77,
    ))
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
    )
    return Components(
        vae=vae,
        text_encoder=text_encoder,
        # Same tokenizer SD 1.5 uses; only a few MB.
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        unet=unet,
        noise_scheduler=DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                                      beta_start=0.00085, beta_end=0.012),
    )
