from torch import nn
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

class ContinuousInstructPix2PixModel(nn.Module):
    """InstructPix2Pix conditioned on both the previous turn and the original image.

    The UNet input is [noisy target | previous turn | original], 12 channels. The
    channels for the original start at zero, so the untrained model matches IP2P.
    """

    def __init__(self, model_id, device, previous_drop_rate, original_drop_rate,
                 instruction_drop_rate, previous_degrade_rate=0.0, model_path=None):
        super().__init__()
        self.device = torch.device(device)
        self.previous_drop_rate = previous_drop_rate
        self.original_drop_rate = original_drop_rate
        self.instruction_drop_rate = instruction_drop_rate
        self.previous_degrade_rate = previous_degrade_rate
        rates = (previous_drop_rate, original_drop_rate, instruction_drop_rate, previous_degrade_rate)
        if not all(0 <= rate <= 1 for rate in rates):
            raise ValueError("Dropout and degrade rates must be between 0 and 1")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        if model_path is None:
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(model_path).to(device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self._expand_conv_in()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

    def _expand_conv_in(self):
        # Accepts SD (4), IP2P (8) or an already-trained continuous (12) UNet.
        old_conv = self.unet.conv_in
        if old_conv.in_channels == 12:
            return
        if old_conv.in_channels not in (4, 8):
            raise ValueError(f"Expected a 4-, 8- or 12-channel UNet, got {old_conv.in_channels}")
        new_conv = nn.Conv2d(
            12, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        self.unet.register_to_config(in_channels=12)

    def encode_condition(self, images):
        # IP2P conditions on the unscaled latent mode, not the scaled diffusion latent.
        return self.vae.encode(images).latent_dist.mode()

    def encode_text(self, instructions):
        input_ids = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(input_ids)[0]

    def decode(self, latents):
        """Scaled diffusion latents -> images in [-1, 1]."""
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def _degrade(self, images):
        # A VAE round trip mimics the detail loss a chained edit accumulates.
        keep = torch.rand(images.shape[0], device=images.device) >= self.previous_degrade_rate
        if keep.all():
            return images
        degraded = images.clone()
        roundtrip = self.vae.decode(self.encode_condition(images[~keep])).sample.clamp(-1, 1)
        degraded[~keep] = roundtrip.to(images.dtype)
        return degraded

    def _drop(self, latents, rate):
        drop = torch.rand(latents.shape[0], device=latents.device) < rate
        return latents.masked_fill(drop[:, None, None, None], 0)

    def forward(self, original, previous, target, instructions):
        """
        original: [B, 3, H, W], the untouched first-turn image, normalized to [-1, 1]
        previous: [B, 3, H, W], the image this turn edits (ground truth or a model rollout)
        target: [B, 3, H, W], normalized to [-1, 1]
        instructions: list[str]

        Returns:
            Scalar diffusion training loss.
        """

        with torch.no_grad():
            if self.training and self.previous_degrade_rate:
                previous = self._degrade(previous)
            previous_latents = self.encode_condition(previous)
            original_latents = self.encode_condition(original)
            target_latents = self.vae.encode(target).latent_dist.sample() * self.vae.config.scaling_factor
            # Each condition drops independently so every guidance term can be estimated.
            if self.training and self.previous_drop_rate:
                previous_latents = self._drop(previous_latents, self.previous_drop_rate)
            if self.training and self.original_drop_rate:
                original_latents = self._drop(original_latents, self.original_drop_rate)
            if self.training and self.instruction_drop_rate:
                instructions = ["" if torch.rand(()).item() < self.instruction_drop_rate else text
                                for text in instructions]
            text_encodings = self.encode_text(instructions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],), device=self.device,
        ).long()
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noisy_latents, previous_latents, original_latents], dim=1)
        model_pred = self.unet(model_input, timesteps, encoder_hidden_states=text_encodings).sample
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            expected = self.noise_scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            expected = noise
        return F.mse_loss(model_pred.float(), expected.float())

    @torch.no_grad()
    def edit(self, original, previous, instructions, num_inference_steps=30, guidance_scale=7.5,
             image_guidance_scale=1.5, original_guidance_scale=1.0, generator=None):
        """Run one editing turn.

        original: [B, 3, H, W] images in [-1, 1]
        previous: [B, 3, H, W] images in [-1, 1], or the latents a previous edit()
            returned, which skips the VAE round trip between turns
        Returns:
            Scaled diffusion latents [B, 4, H/8, W/8]; pass them back as `previous`
            for the next turn, or call decode() to get images.
        """
        dtype = self.unet.dtype
        original_latents = self.encode_condition(original.to(self.device, self.vae.dtype)).to(dtype)
        if previous.shape[1] == 4:
            previous_latents = (previous / self.vae.config.scaling_factor).to(self.device, dtype)
        else:
            previous_latents = self.encode_condition(previous.to(self.device, self.vae.dtype)).to(dtype)
        batch = original_latents.shape[0]
        text = self.encode_text(list(instructions)).to(dtype)
        null_text = self.encode_text([""] * batch).to(dtype)
        zeros = torch.zeros_like(original_latents)

        # Guidance passes, stacked into one batch:
        # (none), (previous), (previous, original), (previous, original, text).
        previous_in = torch.cat([zeros, previous_latents, previous_latents, previous_latents])
        original_in = torch.cat([zeros, zeros, original_latents, original_latents])
        text_in = torch.cat([null_text, null_text, null_text, text])

        scheduler = EulerAncestralDiscreteScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        # Draw on the generator's device in float32, as inference.py does, so a seed
        # gives the same starting noise for both models on any device.
        latents = torch.randn(
            original_latents.shape, generator=generator,
            device=generator.device if generator is not None else self.device, dtype=torch.float32,
        ).to(self.device, dtype) * scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            noisy = scheduler.scale_model_input(torch.cat([latents] * 4), t)
            model_input = torch.cat([noisy, previous_in, original_in], dim=1)
            pred = self.unet(model_input, t, encoder_hidden_states=text_in).sample
            uncond, pred_previous, pred_original, pred_full = pred.chunk(4)
            pred = (uncond
                    + image_guidance_scale * (pred_previous - uncond)
                    + original_guidance_scale * (pred_original - pred_previous)
                    + guidance_scale * (pred_full - pred_original))
            latents = scheduler.step(pred, t, latents, generator=generator).prev_sample
        return latents

    def save_model(self, path):
        self.unet.save_pretrained(path)

    def load(self, path):
        self.unet = UNet2DConditionModel.from_pretrained(path).to(self.device)
        self._expand_conv_in()
        self.unet.requires_grad_(True)
