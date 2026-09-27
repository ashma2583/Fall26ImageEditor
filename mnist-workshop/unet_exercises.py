"""Workshop exercises: timestep conditioning and U-Net skip connections."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map timestep indices (B,) to sinusoidal features (B, dim); dim is even."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    # TODO 4: Finish the sinusoidal timestep embedding.
    # Broadcast t, shape (B,), against frequencies, shape (dim // 2,), to form angles.
    # Concatenate sine and cosine features along the last dimension.
    # The output must have shape (B, dim). Frequencies are provided above.
    raise NotImplementedError("TODO 4: Finish the sinusoidal timestep embedding.")


class ResidualBlock(nn.Module):
    """Two convolutions, a timestep-dependent shift, and a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Start the residual branch at zero: each block first acts like its skip.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # (B, C) -> (B, C, 1, 1): the same shift at every spatial location.
        # TODO 5: Add timestep information to the image features.
        # Apply SiLU to time, then self.time_projection.
        # Reshape (B, C) to (B, C, 1, 1) and add this shift to h.
        raise NotImplementedError("TODO 5: Add timestep information to the image features.")
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Let the 7 × 7 bottleneck features share information across the image."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)  # (B, H*W, C)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)


class UNet(nn.Module):
    """Predict noise with the same (B, 1, 28, 28) shape as the input.

    Spatial path: 28 -> 14 -> 7 -> 14 -> 28. Encoder features are concatenated
    into the decoder at matching resolutions. Each encoder/decoder level has two
    residual blocks. base_channels is a multiple of 8.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        c = base_channels
        time_dim = 4 * c
        self.time_mlp = nn.Sequential(
            nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input_conv = nn.Conv2d(1, c, 3, padding=1)
        self.encoder_28 = ResidualBlock(c, c, time_dim)
        self.down_28 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.encoder_14 = ResidualBlock(c, 2 * c, time_dim)
        self.down_14 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.middle1 = ResidualBlock(2 * c, 2 * c, time_dim)
        self.attention = AttentionBlock(2 * c)
        self.middle2 = ResidualBlock(2 * c, 2 * c, time_dim)
        self.decoder_14 = ResidualBlock(4 * c, 2 * c, time_dim)
        self.decoder_28 = ResidualBlock(3 * c, c, time_dim)
        # Refine features a second time at each encoder/decoder resolution.
        self.encoder_28_extra = ResidualBlock(c, c, time_dim)
        self.encoder_14_extra = ResidualBlock(2 * c, 2 * c, time_dim)
        self.decoder_14_extra = ResidualBlock(2 * c, 2 * c, time_dim)
        self.decoder_28_extra = ResidualBlock(c, c, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x contains noisy images; t contains one integer timestep per image."""
        time = self.time_mlp(timestep_embedding(t, self.base_channels))
        skip_28 = self.encoder_28(self.input_conv(x), time)
        skip_28 = self.encoder_28_extra(skip_28, time)
        skip_14 = self.encoder_14(self.down_28(skip_28), time)
        skip_14 = self.encoder_14_extra(skip_14, time)
        h = self.middle1(self.down_14(skip_14), time)
        h = self.middle2(self.attention(h), time)
        h = F.interpolate(h, size=skip_14.shape[-2:], mode="nearest")
        # TODO 6a: Connect the 14 x 14 encoder skip to the decoder.
        # Concatenate h and skip_14 along the channel dimension (dim=1).
        # Pass the result and time into self.decoder_14, then assign it to h.
        raise NotImplementedError("TODO 6a: Connect the 14 x 14 encoder skip to the decoder.")
        h = self.decoder_14_extra(h, time)
        h = F.interpolate(h, size=skip_28.shape[-2:], mode="nearest")
        # TODO 6b: Connect the 28 x 28 encoder skip to the decoder.
        # Concatenate h and skip_28 along the channel dimension (dim=1).
        # Pass the result and time into self.decoder_28, then assign it to h.
        raise NotImplementedError("TODO 6b: Connect the 28 x 28 encoder skip to the decoder.")
        h = self.decoder_28_extra(h, time)
        return self.output(h)
