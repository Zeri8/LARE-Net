"""HeMIS-SegResNet baseline for arbitrary missing MRI modalities."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.blocks import ConvNormAct3d, DownsampleBlock3d, OutputHead3d, ResidualBlock3d, UpsampleBlock3d


class ModalityEncoder3d(nn.Module):
    """Shared single-modality encoder used for all MRI modalities."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        norm: str = "instance",
        act: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least two levels.")

        self.stem = nn.Sequential(
            ConvNormAct3d(in_channels, channels[0], norm=norm, act=act),
            ResidualBlock3d(channels[0], channels[0], norm=norm, act=act),
        )
        self.down_blocks = nn.ModuleList([
            DownsampleBlock3d(channels[i], channels[i + 1], norm=norm, act=act)
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Tensor with shape [B, 1, D, H, W].

        Returns:
            Feature pyramid from high resolution to low resolution.
        """
        features: List[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        return features


class HeMISFusion(nn.Module):
    """HeMIS mean/variance fusion over available modality features."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, modality_features: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_features: list of four tensors, each [B, C, D, H, W].
            mask: [B, 4], 1 means available.

        Returns:
            Fused feature [B, 2*C, D, H, W] from mean and variance.
        """
        if len(modality_features) != 4:
            raise ValueError(f"Expected 4 modality features, got {len(modality_features)}.")
        if mask.ndim != 2 or mask.shape[1] != 4:
            raise ValueError(f"Expected mask shape [B, 4], got {tuple(mask.shape)}.")

        stacked = torch.stack(modality_features, dim=1)  # [B, 4, C, D, H, W]
        mask = mask.to(device=stacked.device, dtype=stacked.dtype)
        mask = mask[:, :, None, None, None, None]
        count = mask.sum(dim=1).clamp_min(self.eps)

        mean = (stacked * mask).sum(dim=1) / count
        var = (((stacked - mean[:, None]) ** 2) * mask).sum(dim=1) / count
        return torch.cat([mean, var], dim=1)


class HeMISSegResNet(nn.Module):
    """
    HeMIS-SegResNet baseline.

    The model encodes each modality independently using a shared encoder, fuses
    multi-level features with HeMIS mean/variance fusion, and decodes the fused
    feature pyramid using a U-Net-like decoder.
    """

    def __init__(
        self,
        num_modalities: int = 4,
        in_channels_per_modality: int = 1,
        num_classes: int = 3,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        norm: str = "instance",
        act: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if num_modalities != 4:
            raise ValueError("This baseline currently expects exactly four modalities: T1, T1ce, T2, FLAIR.")
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.encoder_channels = tuple(encoder_channels)

        self.encoder = ModalityEncoder3d(
            in_channels=in_channels_per_modality,
            channels=encoder_channels,
            norm=norm,
            act=act,
        )
        self.fusion = HeMISFusion()

        fused_channels = [2 * ch for ch in encoder_channels]
        self.bottleneck = nn.Sequential(
            ResidualBlock3d(fused_channels[-1], fused_channels[-1], norm=norm, act=act),
            ResidualBlock3d(fused_channels[-1], fused_channels[-1], norm=norm, act=act),
        )

        decoder_blocks: List[nn.Module] = []
        in_ch = fused_channels[-1]
        for skip_ch, out_ch in zip(reversed(fused_channels[:-1]), reversed(encoder_channels[:-1])):
            decoder_blocks.append(UpsampleBlock3d(in_ch, skip_ch, out_ch, norm=norm, act=act))
            in_ch = out_ch
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_head = OutputHead3d(in_ch, num_classes)

    def forward(self, image: torch.Tensor, modality_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: Tensor with shape [B, 4, D, H, W].
            modality_mask: Optional tensor [B, 4]. If None, all modalities are assumed available.

        Returns:
            Dict with key ``logits`` of shape [B, 3, D, H, W].
        """
        if image.ndim != 5:
            raise ValueError(f"Expected image shape [B, 4, D, H, W], got {tuple(image.shape)}.")
        if image.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {image.shape[1]}.")

        batch_size = image.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(batch_size, self.num_modalities, device=image.device, dtype=image.dtype)
        else:
            modality_mask = modality_mask.to(device=image.device, dtype=image.dtype)
            if modality_mask.shape != (batch_size, self.num_modalities):
                raise ValueError(f"Expected modality_mask shape {(batch_size, self.num_modalities)}, got {tuple(modality_mask.shape)}.")

        per_modality_pyramids: List[List[torch.Tensor]] = []
        for modality_index in range(self.num_modalities):
            modality_image = image[:, modality_index:modality_index + 1]
            pyramid = self.encoder(modality_image)
            per_modality_pyramids.append(pyramid)

        fused_pyramid: List[torch.Tensor] = []
        num_levels = len(self.encoder_channels)
        for level in range(num_levels):
            level_features = [per_modality_pyramids[m][level] for m in range(self.num_modalities)]
            fused_pyramid.append(self.fusion(level_features, modality_mask))

        x = self.bottleneck(fused_pyramid[-1])
        skips = list(reversed(fused_pyramid[:-1]))
        for block, skip in zip(self.decoder_blocks, skips):
            x = block(x, skip)

        logits = self.output_head(x)
        return {"logits": logits, "modality_mask": modality_mask}


if __name__ == "__main__":
    model = HeMISSegResNet(encoder_channels=(8, 16, 32, 64))
    image = torch.randn(2, 4, 32, 32, 32)
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 0, 1]], dtype=torch.float32)
    output = model(image, mask)
    print(output["logits"].shape)
