"""Reusable 3D network building blocks for baseline models and LARE-Net."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["batch", "instance", "group", "none"]
ActType = Literal["relu", "leaky_relu", "gelu", "silu", "none"]


def get_norm_layer(norm: NormType, num_channels: int, num_groups: int = 8) -> nn.Module:
    """Create a 3D normalization layer."""
    if norm == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


def get_activation(act: ActType) -> nn.Module:
    """Create an activation layer."""
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation type: {act}")


class ConvNormAct3d(nn.Module):
    """3D convolution followed by normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: NormType = "instance",
        act: ActType = "leaky_relu",
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = norm == "none"

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = get_norm_layer(norm, out_channels)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock3d(nn.Module):
    """Simple 3D residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: NormType = "instance",
        act: ActType = "leaky_relu",
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            norm=norm,
            act=act,
        )
        self.conv2 = ConvNormAct3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            norm=norm,
            act="none",
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(norm, out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.out_act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.out_act(out + identity)


class DownsampleBlock3d(nn.Module):
    """Residual downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: NormType = "instance",
        act: ActType = "leaky_relu",
    ) -> None:
        super().__init__()
        self.block = ResidualBlock3d(in_channels, out_channels, stride=2, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock3d(nn.Module):
    """Upsample, concatenate skip features, then refine with residual blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: NormType = "instance",
        act: ActType = "leaky_relu",
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            ResidualBlock3d(out_channels + skip_channels, out_channels, norm=norm, act=act),
            ResidualBlock3d(out_channels, out_channels, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class OutputHead3d(nn.Module):
    """Final 1x1x1 convolution output head."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
