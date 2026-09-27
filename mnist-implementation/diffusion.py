"""The cosine noise schedule and the forward/reverse DDPM equations.

The notebook owns the training and sampling loops. This module only performs
individual diffusion operations, so each equation has one implementation.
"""

import torch


def extract(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Select one coefficient per image and broadcast over (C, H, W)."""
    return values[t][:, None, None, None]


class Diffusion:
    """Indices 0..T-1 refer to successive noising steps.

    Index 0 is already slightly noisy. The clean image lies before index 0.
    Images, timestep tensors, and this schedule must be on the same device.
    """

    def __init__(self, timesteps: int = 1000, device: str | torch.device = "cpu"):
        self.timesteps = timesteps
        # Compute on CPU in float64, then transfer float32 tensors to the device.
        # MPS does not support float64.
        boundaries = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        signal = torch.cos((boundaries + 0.008) / 1.008 * torch.pi / 2).square()
        betas = (1 - signal[1:] / signal[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        self.betas = betas.to(device=device, dtype=torch.float32)
        self.alphas = alphas.to(device=device, dtype=torch.float32)
        self.alpha_bar = alpha_bar.to(device=device, dtype=torch.float32)
        self.posterior_variance = (
            betas * (1 - alpha_bar_previous) / (1 - alpha_bar)
        ).to(device=device, dtype=torch.float32)
        self.clean_weight = (
            betas * alpha_bar_previous.sqrt() / (1 - alpha_bar)
        ).to(device=device, dtype=torch.float32)
        self.noisy_weight = (
            alphas.sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar)
        ).to(device=device, dtype=torch.float32)

    def add_noise(self, clean: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        """Draw x_t directly from x_clean using a supplied Gaussian noise target."""
        alpha_bar = extract(self.alpha_bar, t)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def predict_clean(self, noisy: torch.Tensor, t: torch.Tensor,
                      predicted_noise: torch.Tensor) -> torch.Tensor:
        """Invert the forward equation and keep the estimate in the data range."""
        alpha_bar = extract(self.alpha_bar, t)
        clean = (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        return clean.clamp(-1, 1)

    def denoise_step(self, noisy: torch.Tensor, t: torch.Tensor,
                     predicted_noise: torch.Tensor) -> torch.Tensor:
        """One DDPM step, with fresh noise except at the final step (index 0)."""
        clean = self.predict_clean(noisy, t, predicted_noise)
        mean = extract(self.clean_weight, t) * clean + extract(self.noisy_weight, t) * noisy
        variance = extract(self.posterior_variance, t)
        add_noise = (t > 0)[:, None, None, None]
        return mean + add_noise * variance.sqrt() * torch.randn_like(noisy)
