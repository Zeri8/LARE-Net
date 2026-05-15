"""
BraTS label conversion utilities.

This project trains models with three binary output regions:
    0: WT - Whole Tumor
    1: TC - Tumor Core
    2: ET - Enhancing Tumor

Two common BraTS label encodings are supported:

1. brats_1_2_4
   Raw labels:
       0: background
       1: necrotic / non-enhancing tumor core
       2: edema
       4: enhancing tumor

   Region conversion:
       WT = label in {1, 2, 4}
       TC = label in {1, 4}
       ET = label in {4}

2. brats_1_2_3
   This is used by some preprocessed formats, including certain MONAI examples.
   Raw labels:
       0: background
       1: edema
       2: enhancing tumor
       3: necrotic / non-enhancing tumor core

   Region conversion:
       WT = label in {1, 2, 3}
       TC = label in {2, 3}
       ET = label in {2}

The returned tensor convention is:
    [3, D, H, W] or [B, 3, D, H, W]
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence, Tuple

import torch


LabelMode = Literal["brats_1_2_4", "brats_1_2_3"]
REGION_NAMES: Tuple[str, str, str] = ("WT", "TC", "ET")
NUM_REGIONS: int = len(REGION_NAMES)


def _ensure_long_label(label: torch.Tensor) -> torch.Tensor:
    """Return label tensor as integer class labels."""
    if not torch.is_tensor(label):
        raise TypeError(f"Expected torch.Tensor, got {type(label)}.")
    return label.long()


def _squeeze_single_channel(label: torch.Tensor) -> torch.Tensor:
    """
    Remove a singleton channel dimension if present.

    Accepted shapes:
        [D, H, W]
        [1, D, H, W]
        [B, D, H, W]
        [B, 1, D, H, W]
    """
    if label.ndim == 4 and label.shape[0] == 1:
        # [1, D, H, W] -> [D, H, W]
        return label.squeeze(0)
    if label.ndim == 5 and label.shape[1] == 1:
        # [B, 1, D, H, W] -> [B, D, H, W]
        return label.squeeze(1)
    return label


def validate_label_values(
    label: torch.Tensor,
    mode: LabelMode = "brats_1_2_4",
    allow_unknown: bool = False,
) -> None:
    """
    Validate that label values match the expected BraTS encoding.

    Args:
        label: Raw label tensor.
        mode: Label encoding mode.
        allow_unknown: If True, skip strict value validation.
    """
    if allow_unknown:
        return

    label = _squeeze_single_channel(_ensure_long_label(label))
    unique_values = set(int(v) for v in torch.unique(label).detach().cpu().tolist())

    if mode == "brats_1_2_4":
        valid_values = {0, 1, 2, 4}
    elif mode == "brats_1_2_3":
        valid_values = {0, 1, 2, 3}
    else:
        raise ValueError(f"Unsupported label mode: {mode}.")

    invalid_values = unique_values - valid_values
    if invalid_values:
        raise ValueError(
            f"Found invalid label values {sorted(invalid_values)} for mode '{mode}'. "
            f"Valid values are {sorted(valid_values)}."
        )


def convert_brats_label(
    label: torch.Tensor,
    mode: LabelMode = "brats_1_2_4",
    dtype: torch.dtype = torch.float32,
    validate_values: bool = False,
) -> torch.Tensor:
    """
    Convert raw BraTS labels to three binary region channels: WT, TC, ET.

    Args:
        label: Raw label tensor with shape [D, H, W], [1, D, H, W],
            [B, D, H, W], or [B, 1, D, H, W].
        mode: Raw label encoding mode.
        dtype: Output dtype.
        validate_values: Whether to check raw label values before conversion.

    Returns:
        Tensor with shape [3, D, H, W] for unbatched input, or
        [B, 3, D, H, W] for batched input.
    """
    if validate_values:
        validate_label_values(label, mode=mode)

    label = _squeeze_single_channel(_ensure_long_label(label))

    if label.ndim not in (3, 4):
        raise ValueError(
            "Expected label shape [D, H, W], [1, D, H, W], [B, D, H, W], "
            f"or [B, 1, D, H, W], got {tuple(label.shape)}."
        )

    if mode == "brats_1_2_4":
        wt = (label == 1) | (label == 2) | (label == 4)
        tc = (label == 1) | (label == 4)
        et = label == 4
    elif mode == "brats_1_2_3":
        wt = (label == 1) | (label == 2) | (label == 3)
        tc = (label == 2) | (label == 3)
        et = label == 2
    else:
        raise ValueError(f"Unsupported label mode: {mode}.")

    if label.ndim == 3:
        output = torch.stack([wt, tc, et], dim=0)
    else:
        output = torch.stack([wt, tc, et], dim=1)

    return output.to(dtype=dtype)


def convert_region_to_brats_label(
    regions: torch.Tensor,
    threshold: float = 0.5,
    output_mode: LabelMode = "brats_1_2_4",
) -> torch.Tensor:
    """
    Convert three region channels WT/TC/ET back to a raw BraTS-like label map.

    Args:
        regions: Tensor with shape [3, D, H, W] or [B, 3, D, H, W].
            Channel order must be WT, TC, ET.
        threshold: Threshold for binarizing probabilistic region maps.
        output_mode: Desired raw label encoding.

    Returns:
        Raw label map with shape [D, H, W] or [B, D, H, W].

    Notes:
        BraTS regions are nested but model outputs may violate nesting.
        This function resolves labels with priority ET > TC > WT.
    """
    if regions.ndim not in (4, 5):
        raise ValueError(f"Expected regions shape [3, D, H, W] or [B, 3, D, H, W], got {tuple(regions.shape)}.")

    if regions.ndim == 4:
        if regions.shape[0] != NUM_REGIONS:
            raise ValueError(f"Expected {NUM_REGIONS} channels, got {regions.shape[0]}.")
        batched = False
        region_mask = regions.unsqueeze(0) > threshold
    else:
        if regions.shape[1] != NUM_REGIONS:
            raise ValueError(f"Expected {NUM_REGIONS} channels, got {regions.shape[1]}.")
        batched = True
        region_mask = regions > threshold

    wt = region_mask[:, 0]
    tc = region_mask[:, 1]
    et = region_mask[:, 2]

    label = torch.zeros_like(wt, dtype=torch.long)

    if output_mode == "brats_1_2_4":
        # WT-only area is edema -> 2. TC area defaults to necrotic/non-enhancing core -> 1.
        label[wt] = 2
        label[tc] = 1
        label[et] = 4
    elif output_mode == "brats_1_2_3":
        # WT-only area is edema -> 1. TC area defaults to necrotic/non-enhancing core -> 3.
        label[wt] = 1
        label[tc] = 3
        label[et] = 2
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}.")

    if not batched:
        label = label.squeeze(0)

    return label


def enforce_region_nesting(regions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Enforce BraTS region nesting: ET is inside TC, and TC is inside WT.

    Args:
        regions: Tensor with shape [3, D, H, W] or [B, 3, D, H, W].
        threshold: Threshold used to binarize region maps.

    Returns:
        Binary tensor with the same shape as regions.
    """
    if regions.ndim not in (4, 5):
        raise ValueError(f"Expected regions shape [3, D, H, W] or [B, 3, D, H, W], got {tuple(regions.shape)}.")

    binary = regions > threshold

    if regions.ndim == 4:
        wt = binary[0] | binary[1] | binary[2]
        tc = binary[1] | binary[2]
        et = binary[2]
        nested = torch.stack([wt, tc, et], dim=0)
    else:
        wt = binary[:, 0] | binary[:, 1] | binary[:, 2]
        tc = binary[:, 1] | binary[:, 2]
        et = binary[:, 2]
        nested = torch.stack([wt, tc, et], dim=1)

    return nested.to(dtype=regions.dtype)


def get_region_names() -> Tuple[str, str, str]:
    """Return the output region names in channel order."""
    return REGION_NAMES


def region_channel_index(region_name: str) -> int:
    """Return channel index for a region name."""
    normalized = region_name.strip().upper()
    if normalized not in REGION_NAMES:
        valid = ", ".join(REGION_NAMES)
        raise ValueError(f"Unknown region '{region_name}'. Valid regions: {valid}.")
    return REGION_NAMES.index(normalized)


if __name__ == "__main__":
    demo_label = torch.zeros(8, 8, 8, dtype=torch.long)
    demo_label[1:5, 1:5, 1:5] = 2
    demo_label[2:4, 2:4, 2:4] = 1
    demo_label[3:4, 3:4, 3:4] = 4

    regions = convert_brats_label(demo_label, mode="brats_1_2_4", validate_values=True)
    restored = convert_region_to_brats_label(regions, output_mode="brats_1_2_4")

    print("Raw label shape:", tuple(demo_label.shape))
    print("Region shape:", tuple(regions.shape))
    print("Restored label shape:", tuple(restored.shape))
    print("Region voxel counts:", {name: int(regions[i].sum().item()) for i, name in enumerate(REGION_NAMES)})
