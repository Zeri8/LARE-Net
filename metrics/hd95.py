"""HD95 metric utilities for binary 3D segmentation masks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


REGION_NAMES: Tuple[str, str, str] = ("WT", "TC", "ET")


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return binary surface voxels of a 3D mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded


def binary_hd95(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, float, float] | None = None) -> float:
    """
    Compute 95th percentile Hausdorff distance for one binary 3D mask.

    If both prediction and target are empty, returns 0. If only one is empty,
    returns inf because the boundary distance is undefined and represents failure.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_target = distance_transform_edt(~target_surface, sampling=spacing)

    distances_pred_to_target = dt_target[pred_surface]
    distances_target_to_pred = dt_pred[target_surface]

    distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def hd95_per_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    spacing: Tuple[float, float, float] | None = None,
    empty_value: float = 373.13,
) -> torch.Tensor:
    """
    Compute mean HD95 per channel over a batch.

    Args:
        prediction: [B, 3, D, H, W], logits or probabilities.
        target: [B, 3, D, H, W], binary target.
        threshold: Probability threshold.
        from_logits: If True, apply sigmoid to prediction.
        spacing: Optional voxel spacing.
        empty_value: Replacement value when HD95 is inf due to one empty mask.

    Returns:
        Tensor [3] with mean HD95 for WT, TC, ET.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}.")
    if prediction.ndim != 5 or prediction.shape[1] != len(REGION_NAMES):
        raise ValueError(f"Expected prediction shape [B, 3, D, H, W], got {tuple(prediction.shape)}.")

    probs = torch.sigmoid(prediction) if from_logits else prediction
    pred = (probs > threshold).detach().cpu().numpy().astype(bool)
    tgt = (target > 0.5).detach().cpu().numpy().astype(bool)

    values = np.zeros((prediction.shape[0], len(REGION_NAMES)), dtype=np.float32)
    for b in range(prediction.shape[0]):
        for c in range(len(REGION_NAMES)):
            value = binary_hd95(pred[b, c], tgt[b, c], spacing=spacing)
            if np.isinf(value) or np.isnan(value):
                value = empty_value
            values[b, c] = value

    return torch.from_numpy(values.mean(axis=0))


def hd95_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    spacing: Tuple[float, float, float] | None = None,
) -> Dict[str, float]:
    """Return HD95 as a Python dictionary."""
    values = hd95_per_channel(prediction, target, threshold=threshold, from_logits=from_logits, spacing=spacing)
    output = {name: float(values[index]) for index, name in enumerate(REGION_NAMES)}
    output["Avg"] = float(values.mean())
    return output
