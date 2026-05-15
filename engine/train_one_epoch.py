"""One-epoch training loop for baseline models."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.modality_mask import sample_batch_masks


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = True,
    grad_clip: Optional[float] = None,
    min_modalities: int = 1,
    max_modalities: int = 4,
) -> Dict[str, float]:
    """Train one epoch with random modality masks."""
    model.train()
    scaler = GradScaler(enabled=amp)
    running: Dict[str, float] = {"loss": 0.0, "dice_loss": 0.0, "bce_loss": 0.0}
    num_steps = 0

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch in progress:
        image = batch["image"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.float32)
        mask = sample_batch_masks(
            batch_size=image.shape[0],
            min_modalities=min_modalities,
            max_modalities=max_modalities,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            output = model(image, mask)
            losses = criterion(output["logits"], label)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        num_steps += 1
        for key in running:
            if key in losses:
                running[key] += float(losses[key].detach().cpu())
        progress.set_postfix(loss=running["loss"] / num_steps)

    return {key: value / max(num_steps, 1) for key, value in running.items()}
