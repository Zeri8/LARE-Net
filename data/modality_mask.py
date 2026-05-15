"""
Utilities for arbitrary missing MRI modality handling.

Modality order is fixed as:
    0: T1
    1: T1ce
    2: T2
    3: FLAIR

Input tensor convention:
    image: [B, 4, D, H, W]

Modality mask convention:
    mask: [B, 4]

where 1 means the modality is available and 0 means the modality is missing.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import torch


MODALITY_NAMES: Tuple[str, str, str, str] = ("T1", "T1ce", "T2", "FLAIR")
NUM_MODALITIES: int = len(MODALITY_NAMES)
ModalityMask = Tuple[int, int, int, int]


def get_all_modality_masks() -> List[ModalityMask]:
    """
    Return all 15 non-empty modality masks for four MRI modalities.

    Examples:
        (1, 0, 0, 0): only T1 is available.
        (0, 1, 0, 1): T1ce and FLAIR are available.
        (1, 1, 1, 1): all modalities are available.
    """
    masks: List[ModalityMask] = []
    modality_indices = list(range(NUM_MODALITIES))

    for num_available in range(1, NUM_MODALITIES + 1):
        for selected in combinations(modality_indices, num_available):
            mask = [0] * NUM_MODALITIES
            for index in selected:
                mask[index] = 1
            masks.append(tuple(mask))  # type: ignore[arg-type]

    return masks


ALL_MODALITY_MASKS: List[ModalityMask] = get_all_modality_masks()


def validate_mask(mask: Sequence[int]) -> None:
    """
    Validate a single modality mask.

    Args:
        mask: Sequence with four binary values.

    Raises:
        ValueError: If the mask is malformed or empty.
    """
    if len(mask) != NUM_MODALITIES:
        raise ValueError(f"Expected mask length {NUM_MODALITIES}, got {len(mask)}.")

    values = [int(value) for value in mask]
    if any(value not in (0, 1) for value in values):
        raise ValueError(f"Modality mask must be binary, got {mask}.")

    if sum(values) == 0:
        raise ValueError("Empty modality mask is invalid. At least one modality must be available.")


def mask_to_name(mask: Sequence[int]) -> str:
    """
    Convert a modality mask to a readable name.

    Examples:
        [1, 0, 0, 0] -> "T1"
        [0, 1, 0, 1] -> "T1ce+FLAIR"
        [1, 1, 1, 1] -> "T1+T1ce+T2+FLAIR"
    """
    validate_mask(mask)
    names = [name for name, value in zip(MODALITY_NAMES, mask) if int(value) == 1]
    return "+".join(names)


def name_to_mask(name: str) -> ModalityMask:
    """
    Convert a modality-combination name to a binary mask.

    Examples:
        "T1" -> (1, 0, 0, 0)
        "T1ce+FLAIR" -> (0, 1, 0, 1)
        "T1+T1ce+T2+FLAIR" -> (1, 1, 1, 1)
    """
    parts = [part.strip() for part in name.split("+") if part.strip()]
    if not parts:
        raise ValueError("Empty modality name is invalid.")

    modality_to_index: Dict[str, int] = {modality: index for index, modality in enumerate(MODALITY_NAMES)}
    mask = [0] * NUM_MODALITIES

    for part in parts:
        if part not in modality_to_index:
            valid = ", ".join(MODALITY_NAMES)
            raise ValueError(f"Unknown modality '{part}'. Valid modalities: {valid}.")
        mask[modality_to_index[part]] = 1

    validate_mask(mask)
    return tuple(mask)  # type: ignore[return-value]


def sample_random_mask(
    min_modalities: int = 1,
    max_modalities: int = NUM_MODALITIES,
    probabilities: Optional[Sequence[float]] = None,
) -> ModalityMask:
    """
    Randomly sample one valid modality mask.

    Args:
        min_modalities: Minimum number of available modalities.
        max_modalities: Maximum number of available modalities.
        probabilities: Optional sampling probabilities over filtered candidate masks.

    Returns:
        A modality mask, for example (1, 0, 1, 0).
    """
    if min_modalities < 1:
        raise ValueError("min_modalities must be at least 1.")
    if max_modalities > NUM_MODALITIES:
        raise ValueError(f"max_modalities cannot exceed {NUM_MODALITIES}.")
    if min_modalities > max_modalities:
        raise ValueError("min_modalities cannot be greater than max_modalities.")

    candidates = [
        mask for mask in ALL_MODALITY_MASKS
        if min_modalities <= sum(mask) <= max_modalities
    ]

    if probabilities is not None:
        if len(probabilities) != len(candidates):
            raise ValueError(f"Expected {len(candidates)} probabilities, got {len(probabilities)}.")
        return random.choices(candidates, weights=probabilities, k=1)[0]

    return random.choice(candidates)


def sample_batch_masks(
    batch_size: int,
    min_modalities: int = 1,
    max_modalities: int = NUM_MODALITIES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample one random modality mask for each item in a batch.

    Returns:
        Tensor with shape [B, 4] and dtype float32.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    masks = [
        sample_random_mask(min_modalities=min_modalities, max_modalities=max_modalities)
        for _ in range(batch_size)
    ]
    return torch.tensor(masks, dtype=torch.float32, device=device)


def get_fixed_mask_batch(
    mask: Sequence[int],
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Repeat one modality mask for a full batch.

    This is useful for validation and testing under a fixed modality combination.
    """
    validate_mask(mask)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    return mask_tensor.unsqueeze(0).repeat(batch_size, 1)


def apply_modality_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a modality mask to a dense four-channel MRI tensor.

    Args:
        image: Tensor with shape [B, 4, D, H, W].
        mask: Tensor with shape [B, 4].

    Returns:
        Tensor with the same shape as image. Missing modalities are set to zero.

    Notes:
        This function is mainly intended for Modality Dropout and simple baselines.
        LARE-Net modules should use the mask internally instead of relying only on
        zero-filled inputs.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected image shape [B, 4, D, H, W], got {tuple(image.shape)}.")
    if image.shape[1] != NUM_MODALITIES:
        raise ValueError(f"Expected {NUM_MODALITIES} modalities, got {image.shape[1]}.")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape [B, 4], got {tuple(mask.shape)}.")
    if mask.shape[0] != image.shape[0]:
        raise ValueError(f"Batch size mismatch: image has {image.shape[0]}, mask has {mask.shape[0]}.")
    if mask.shape[1] != NUM_MODALITIES:
        raise ValueError(f"Expected mask shape [B, {NUM_MODALITIES}], got {tuple(mask.shape)}.")

    mask = mask.to(device=image.device, dtype=image.dtype)
    mask = mask[:, :, None, None, None]
    return image * mask


def split_available_modalities(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> List[List[Tuple[int, torch.Tensor]]]:
    """
    Split a dense four-channel batch into available modality tensors per sample.

    Args:
        image: Tensor with shape [B, 4, D, H, W].
        mask: Tensor with shape [B, 4].

    Returns:
        A nested list where:
            batch_modalities[b] = [(modality_index, modality_tensor), ...]

        Each modality_tensor has shape [1, D, H, W].
    """
    if image.ndim != 5:
        raise ValueError(f"Expected image shape [B, 4, D, H, W], got {tuple(image.shape)}.")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape [B, 4], got {tuple(mask.shape)}.")
    if image.shape[0] != mask.shape[0]:
        raise ValueError("Batch size mismatch between image and mask.")
    if image.shape[1] != NUM_MODALITIES or mask.shape[1] != NUM_MODALITIES:
        raise ValueError("Expected four modalities in image and mask.")

    batch_modalities: List[List[Tuple[int, torch.Tensor]]] = []

    for batch_index in range(image.shape[0]):
        sample_modalities: List[Tuple[int, torch.Tensor]] = []
        for modality_index in range(NUM_MODALITIES):
            if int(mask[batch_index, modality_index].item()) == 1:
                sample_modalities.append((modality_index, image[batch_index, modality_index:modality_index + 1]))

        if not sample_modalities:
            raise ValueError(f"Sample {batch_index} has no available modality.")

        batch_modalities.append(sample_modalities)

    return batch_modalities


def describe_all_masks() -> List[str]:
    """
    Return readable descriptions for all 15 non-empty masks.
    """
    return [f"{mask}: {mask_to_name(mask)}" for mask in ALL_MODALITY_MASKS]


def get_single_modality_masks() -> List[ModalityMask]:
    """Return the four single-modality masks."""
    return [mask for mask in ALL_MODALITY_MASKS if sum(mask) == 1]


def get_two_modality_masks() -> List[ModalityMask]:
    """Return all six two-modality masks."""
    return [mask for mask in ALL_MODALITY_MASKS if sum(mask) == 2]


def get_three_modality_masks() -> List[ModalityMask]:
    """Return all four three-modality masks."""
    return [mask for mask in ALL_MODALITY_MASKS if sum(mask) == 3]


def get_full_modality_mask() -> ModalityMask:
    """Return the complete four-modality mask."""
    return (1, 1, 1, 1)


if __name__ == "__main__":
    print("All modality masks:")
    for description in describe_all_masks():
        print(description)

    demo_image = torch.randn(2, 4, 8, 8, 8)
    demo_mask = sample_batch_masks(batch_size=2)
    demo_masked_image = apply_modality_mask(demo_image, demo_mask)

    print("Demo image shape:", tuple(demo_image.shape))
    print("Demo mask:", demo_mask)
    print("Demo masked image shape:", tuple(demo_masked_image.shape))
