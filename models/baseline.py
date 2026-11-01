from torch import nn
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

class InstructPix2PixModel(nn.Module):
    def __init__(self, model_id, device, source_drop_rate, instruction_drop_rate):
        super().__init__()
        self.device = torch.device(device)
        self.source_drop_rate = source_drop_rate
        self.instruction_drop_rate = instruction_drop_rate
        if not 0 <= source_drop_rate <= 1 or not 0 <= instruction_drop_rate <= 1:
            raise ValueError("Condition dropout rates must be between 0 and 1")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        old_conv = self.unet.conv_in
        if old_conv.in_channels == 4:
            new_conv = nn.Conv2d(
                8, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :4].copy_(old_conv.weight)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.unet.conv_in = new_conv
            self.unet.register_to_config(in_channels=8)
        elif old_conv.in_channels != 8:
            raise ValueError(f"Expected a 4- or 8-channel UNet, got {old_conv.in_channels}")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

    def forward(self, source, target, instructions):
        """
        source: [B, 3, H, W], normalized to [-1, 1]
        target: [B, 3, H, W], normalized to [-1, 1]
        instructions: list[str]

        Returns:
            Scalar diffusion training loss.
        """

        with torch.no_grad():
            source_latents = self.vae.encode(source).latent_dist.sample() * self.vae.config.scaling_factor
            target_latents = self.vae.encode(target).latent_dist.sample() * self.vae.config.scaling_factor
            if self.training and self.source_drop_rate:
                drop_source = torch.rand(source.shape[0], device=source.device) < self.source_drop_rate
                source_latents = source_latents.masked_fill(drop_source[:, None, None, None], 0)
            if self.training and self.instruction_drop_rate:
                instructions = ["" if torch.rand(()).item() < self.instruction_drop_rate else text
                                for text in instructions]
            input_ids = self.tokenizer(
                instructions,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(self.device)
            text_encodings = self.text_encoder(input_ids)[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],), device=self.device,
        ).long()
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noisy_latents, source_latents], dim=1)
        model_pred = self.unet(model_input, timesteps, encoder_hidden_states=text_encodings).sample
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            expected = self.noise_scheduler.get_velocity(target_latents, noise, timesteps)
        else:
            expected = noise
        return F.mse_loss(model_pred.float(), expected.float())

    def save_model(self, path):
        self.unet.save_pretrained(path)

    def load(self, path):
        from diffusers import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(
            path
        ).to(self.device)

        self.unet.requires_grad_(True)
