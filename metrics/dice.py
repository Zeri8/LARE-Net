"""Dice metrics for multi-label BraTS region segmentation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


REGION_NAMES: Tuple[str, str, str] = ("WT", "TC", "ET")


def dice_per_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Compute hard Dice per channel.

    Args:
        prediction: [B, 3, D, H, W], logits or probabilities.
        target: [B, 3, D, H, W], binary target.
        threshold: Probability threshold.
        eps: Numerical stability constant.
        from_logits: If True, apply sigmoid to prediction.

    Returns:
        Tensor [3] with Dice for WT, TC, ET.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}.")
    if prediction.ndim != 5 or prediction.shape[1] != len(REGION_NAMES):
        raise ValueError(f"Expected prediction shape [B, 3, D, H, W], got {tuple(prediction.shape)}.")

    probs = torch.sigmoid(prediction) if from_logits else prediction
    pred = (probs > threshold).to(dtype=target.dtype)
    target = target.to(dtype=pred.dtype)

    dims = (0, 2, 3, 4)
    intersection = torch.sum(pred * target, dim=dims)
    pred_sum = torch.sum(pred, dim=dims)
    target_sum = torch.sum(target, dim=dims)
    return (2.0 * intersection + eps) / (pred_sum + target_sum + eps)


def dice_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Dict[str, float]:
    """Return Dice as a Python dictionary."""
    dice = dice_per_channel(prediction, target, threshold=threshold, from_logits=from_logits)
    output = {name: float(dice[index].detach().cpu()) for index, name in enumerate(REGION_NAMES)}
    output["Avg"] = float(dice.mean().detach().cpu())
    return output
