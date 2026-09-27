"""Workshop exercises: cosine scheduling and DDPM equations.

The notebook owns the training and sampling loops. This module only performs
individual diffusion operations, so each equation has one implementation.
"""

import torch


def extract(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Select one coefficient per image and broadcast over (C, H, W)."""
    # TODO 2: Select and reshape one coefficient per image.
    # values: (T,), t: (B,). Return a tensor of shape (B, 1, 1, 1).
    # Index with t, then add three singleton dimensions.
    raise NotImplementedError("TODO 2: Select and reshape one coefficient per image.")


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
        # TODO 1: Compute the one-step and cumulative signal retention.
        # betas has shape (T,). Use alpha_t = 1 - beta_t.
        # Use torch.cumprod along the timestep dimension to obtain alpha_bar.
        raise NotImplementedError("TODO 1: Compute the one-step and cumulative signal retention.")
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
        # TODO 3: Sample the noisy image directly at timestep t.
        # Use extract to gather alpha_bar for every image.
        # Scale clean by sqrt(alpha_bar) and noise by sqrt(1 - alpha_bar).
        # Return their sum. Keep the supplied noise; it is the training target.
        raise NotImplementedError("TODO 3: Sample the noisy image directly at timestep t.")

    def predict_clean(self, noisy: torch.Tensor, t: torch.Tensor,
                      predicted_noise: torch.Tensor) -> torch.Tensor:
        """Invert the forward equation and keep the estimate in the data range."""
        # TODO 8: Estimate the clean image from predicted noise.
        # Rearrange the forward noising equation to solve for the clean image.
        # Gather alpha_bar using extract and clamp the result to [-1, 1].
        raise NotImplementedError("TODO 8: Estimate the clean image from predicted noise.")

    def denoise_step(self, noisy: torch.Tensor, t: torch.Tensor,
                     predicted_noise: torch.Tensor) -> torch.Tensor:
        """One DDPM step, with fresh noise except at the final step (index 0)."""
        clean = self.predict_clean(noisy, t, predicted_noise)
        # TODO 9: Perform one DDPM reverse step.
        # Form the mean using clean_weight * clean + noisy_weight * noisy.
        # Gather posterior_variance and add fresh Gaussian noise scaled by its square root.
        # Use a (B, 1, 1, 1) mask so samples with t == 0 receive no extra noise.
        # Return the previous image. Coefficients have already been precomputed.
        raise NotImplementedError("TODO 9: Perform one DDPM reverse step.")
