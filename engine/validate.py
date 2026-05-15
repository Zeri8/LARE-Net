"""Validation loop for baseline models."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from data.modality_mask import get_fixed_mask_batch, sample_batch_masks
from metrics.dice import dice_dict
from metrics.hd95 import hd95_dict


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    amp: bool = True,
    fixed_mask: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
    compute_hd95: bool = False,
) -> Dict[str, float]:
    """Validate a model under either random masks or one fixed mask."""
    model.eval()
    loss_total = 0.0
    dice_sums = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Avg": 0.0}
    hd95_sums = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Avg": 0.0}
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", leave=False):
            image = batch["image"].to(device=device, dtype=torch.float32)
            label = batch["label"].to(device=device, dtype=torch.float32)
            if fixed_mask is None:
                mask = sample_batch_masks(batch_size=image.shape[0], device=device)
            else:
                mask = get_fixed_mask_batch(fixed_mask, batch_size=image.shape[0], device=device)

            with autocast(enabled=amp):
                output = model(image, mask)
                logits = output["logits"]
                if criterion is not None:
                    losses = criterion(logits, label)
                    loss_total += float(losses["loss"].detach().cpu())

            d = dice_dict(logits, label, threshold=threshold, from_logits=True)
            for key in dice_sums:
                dice_sums[key] += d[key]

            if compute_hd95:
                h = hd95_dict(logits, label, threshold=threshold, from_logits=True)
                for key in hd95_sums:
                    hd95_sums[key] += h[key]

            steps += 1

    metrics: Dict[str, float] = {}
    if criterion is not None:
        metrics["loss"] = loss_total / max(steps, 1)
    for key, value in dice_sums.items():
        metrics[f"dice_{key}"] = value / max(steps, 1)
    if compute_hd95:
        for key, value in hd95_sums.items():
            metrics[f"hd95_{key}"] = value / max(steps, 1)
    return metrics
