"""
Dice + BCE losses for BraTS region-based segmentation.

This project predicts three overlapping BraTS regions:
    0: WT - Whole Tumor
    1: TC - Tumor Core
    2: ET - Enhancing Tumor

These regions are not mutually exclusive, so the model should use sigmoid outputs
and multi-label binary losses instead of softmax cross-entropy.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REGION_NAMES: Tuple[str, str, str] = ("WT", "TC", "ET")


def _validate_prediction_and_target(logits: torch.Tensor, target: torch.Tensor) -> None:
    """Validate segmentation prediction and target tensors."""
    if logits.ndim != 5:
        raise ValueError(f"Expected logits shape [B, C, D, H, W], got {tuple(logits.shape)}.")
    if target.ndim != 5:
        raise ValueError(f"Expected target shape [B, C, D, H, W], got {tuple(target.shape)}.")
    if logits.shape != target.shape:
        raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target.shape)}.")
    if logits.shape[1] != len(REGION_NAMES):
        raise ValueError(f"Expected {len(REGION_NAMES)} output channels, got {logits.shape[1]}.")


def dice_score_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute hard Dice score per channel from logits.

    Args:
        logits: Model output logits with shape [B, 3, D, H, W].
        target: Binary target with shape [B, 3, D, H, W].
        threshold: Sigmoid probability threshold.
        eps: Numerical stability constant.

    Returns:
        Tensor with shape [3], containing mean Dice for WT, TC, ET.
    """
    _validate_prediction_and_target(logits, target)

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).to(dtype=target.dtype)
    target = target.to(dtype=pred.dtype)

    reduce_dims = (0, 2, 3, 4)
    intersection = torch.sum(pred * target, dim=reduce_dims)
    pred_sum = torch.sum(pred, dim=reduce_dims)
    target_sum = torch.sum(target, dim=reduce_dims)

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice


def soft_dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    squared_denominator: bool = False,
) -> torch.Tensor:
    """
    Compute soft Dice score per channel from logits.

    Args:
        logits: Model output logits with shape [B, 3, D, H, W].
        target: Binary target with shape [B, 3, D, H, W].
        eps: Numerical stability constant.
        squared_denominator: If True, use squared terms in the denominator.

    Returns:
        Tensor with shape [3], containing mean soft Dice for WT, TC, ET.
    """
    _validate_prediction_and_target(logits, target)

    probs = torch.sigmoid(logits)
    target = target.to(dtype=probs.dtype)

    reduce_dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target, dim=reduce_dims)

    if squared_denominator:
        denominator = torch.sum(probs * probs, dim=reduce_dims) + torch.sum(target * target, dim=reduce_dims)
    else:
        denominator = torch.sum(probs, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice


class SoftDiceLoss(nn.Module):
    """
    Multi-label soft Dice loss for WT/TC/ET segmentation.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        squared_denominator: bool = False,
        channel_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.squared_denominator = squared_denominator

        if channel_weights is not None:
            if len(channel_weights) != len(REGION_NAMES):
                raise ValueError(f"Expected {len(REGION_NAMES)} channel weights, got {len(channel_weights)}.")
            self.register_buffer("channel_weights", torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar soft Dice loss."""
        dice = soft_dice_score(
            logits=logits,
            target=target,
            eps=self.eps,
            squared_denominator=self.squared_denominator,
        )
        loss_per_channel = 1.0 - dice

        if self.channel_weights is not None:
            weights = self.channel_weights.to(device=logits.device, dtype=loss_per_channel.dtype)
            loss = torch.sum(loss_per_channel * weights) / weights.sum().clamp_min(self.eps)
        else:
            loss = loss_per_channel.mean()

        return loss


class DiceBCELoss(nn.Module):
    """
    Combined soft Dice loss and BCEWithLogits loss.

    This is the default segmentation loss for HeMIS-SegResNet and LARE-Net.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        eps: float = 1e-6,
        squared_denominator: bool = False,
        channel_weights: Optional[Sequence[float]] = None,
        pos_weight: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or bce_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if dice_weight == 0 and bce_weight == 0:
            raise ValueError("At least one of dice_weight or bce_weight must be positive.")

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = SoftDiceLoss(
            eps=eps,
            squared_denominator=squared_denominator,
            channel_weights=channel_weights,
        )

        if pos_weight is not None:
            if len(pos_weight) != len(REGION_NAMES):
                raise ValueError(f"Expected {len(REGION_NAMES)} positive weights, got {len(pos_weight)}.")
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def _bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)

        if self.pos_weight is None:
            return F.binary_cross_entropy_with_logits(logits, target)

        # BCEWithLogitsLoss expects pos_weight to be broadcastable to logits.
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        pos_weight = pos_weight.view(1, -1, 1, 1, 1)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model output logits with shape [B, 3, D, H, W].
            target: Binary target with shape [B, 3, D, H, W].

        Returns:
            Dictionary containing total loss and individual components.
        """
        _validate_prediction_and_target(logits, target)

        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self.dice_weight > 0:
            dice = self.dice_loss(logits, target)
            losses["dice_loss"] = dice
            total = total + self.dice_weight * dice
        else:
            losses["dice_loss"] = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self.bce_weight > 0:
            bce = self._bce_loss(logits, target)
            losses["bce_loss"] = bce
            total = total + self.bce_weight * bce
        else:
            losses["bce_loss"] = torch.zeros((), device=logits.device, dtype=logits.dtype)

        losses["loss"] = total
        return losses


def build_segmentation_loss(
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    channel_weights: Optional[Sequence[float]] = None,
    pos_weight: Optional[Sequence[float]] = None,
) -> DiceBCELoss:
    """
    Convenience factory for the default segmentation loss.
    """
    return DiceBCELoss(
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        channel_weights=channel_weights,
        pos_weight=pos_weight,
    )


if __name__ == "__main__":
    demo_logits = torch.randn(2, 3, 16, 16, 16)
    demo_target = (torch.rand(2, 3, 16, 16, 16) > 0.8).float()

    criterion = DiceBCELoss()
    output = criterion(demo_logits, demo_target)
    dice = dice_score_from_logits(demo_logits, demo_target)

    print("Loss dict:", {key: float(value.detach().cpu()) for key, value in output.items()})
    print("Dice:", {name: float(dice[index].detach().cpu()) for index, name in enumerate(REGION_NAMES)})
