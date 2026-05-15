"""BraTS dataset utilities for baseline training.

Expected case layout examples:

1. BraTS-style folder per case:
   root/BraTS-XXXX/
       BraTS-XXXX_t1.nii.gz
       BraTS-XXXX_t1ce.nii.gz
       BraTS-XXXX_t2.nii.gz
       BraTS-XXXX_flair.nii.gz
       BraTS-XXXX_seg.nii.gz

2. Alternative names are also matched by suffix keywords.

The dataset returns:
    image: [4, D, H, W]
    label: [3, D, H, W]  (WT, TC, ET)
    case_id: str
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from data.label_convert import convert_brats_label


MODALITY_KEYS: Tuple[str, str, str, str] = ("t1", "t1ce", "t2", "flair")


def _load_nifti(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    return array


def _zscore_nonzero(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return volume.astype(np.float32)
    values = volume[mask]
    mean = values.mean()
    std = values.std()
    volume = (volume - mean) / (std + eps)
    volume[~mask] = 0.0
    return volume.astype(np.float32)


def _find_file(case_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    files = [p for p in case_dir.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
    lower_map = {p.name.lower(): p for p in files}
    for name in lower_map:
        for candidate in candidates:
            if candidate in name:
                return lower_map[name]
    return None


def discover_brats_cases(root: str | Path) -> List[Dict[str, str]]:
    """Discover BraTS cases from a root directory."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    cases: List[Dict[str, str]] = []

    for case_dir in sorted(case_dirs):
        t1 = _find_file(case_dir, ["_t1.", "-t1.", "t1.nii"])
        t1ce = _find_file(case_dir, ["_t1ce.", "-t1ce.", "t1ce.nii", "_t1gd.", "-t1gd."])
        t2 = _find_file(case_dir, ["_t2.", "-t2.", "t2.nii"])
        flair = _find_file(case_dir, ["_flair.", "-flair.", "flair.nii"])
        seg = _find_file(case_dir, ["_seg.", "-seg.", "seg.nii", "label", "mask"])

        if all(path is not None for path in (t1, t1ce, t2, flair, seg)):
            cases.append({
                "case_id": case_dir.name,
                "t1": str(t1),
                "t1ce": str(t1ce),
                "t2": str(t2),
                "flair": str(flair),
                "seg": str(seg),
            })

    if not cases:
        raise RuntimeError(f"No complete BraTS cases found under {root}")
    return cases


class BraTSDataset(Dataset):
    """Simple BraTS dataset for baseline experiments."""

    def __init__(
        self,
        root: str | Path,
        label_mode: str = "brats_1_2_4",
        normalize: bool = True,
        crop_size: Optional[Tuple[int, int, int]] = None,
        random_crop: bool = False,
    ) -> None:
        self.root = Path(root)
        self.cases = discover_brats_cases(self.root)
        self.label_mode = label_mode
        self.normalize = normalize
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.cases)

    def _crop(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return image, label
        _, d, h, w = image.shape
        cd, ch, cw = self.crop_size
        if d < cd or h < ch or w < cw:
            pad_d = max(cd - d, 0)
            pad_h = max(ch - h, 0)
            pad_w = max(cw - w, 0)
            pad = (0, pad_w, 0, pad_h, 0, pad_d)
            image = torch.nn.functional.pad(image, pad)
            label = torch.nn.functional.pad(label, pad)
            _, d, h, w = image.shape

        if self.random_crop:
            sd = torch.randint(0, d - cd + 1, (1,)).item()
            sh = torch.randint(0, h - ch + 1, (1,)).item()
            sw = torch.randint(0, w - cw + 1, (1,)).item()
        else:
            sd = max((d - cd) // 2, 0)
            sh = max((h - ch) // 2, 0)
            sw = max((w - cw) // 2, 0)

        image = image[:, sd:sd + cd, sh:sh + ch, sw:sw + cw]
        label = label[:, sd:sd + cd, sh:sh + ch, sw:sw + cw]
        return image, label

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        case = self.cases[index]
        modalities = []
        for key in MODALITY_KEYS:
            volume = _load_nifti(Path(case[key]))
            if self.normalize:
                volume = _zscore_nonzero(volume)
            modalities.append(torch.from_numpy(volume).float())

        image = torch.stack(modalities, dim=0)  # [4, D, H, W]
        raw_label = torch.from_numpy(_load_nifti(Path(case["seg"]))).long()
        label = convert_brats_label(raw_label, mode=self.label_mode, dtype=torch.float32)
        image, label = self._crop(image, label)
        return {"image": image, "label": label, "case_id": case["case_id"]}
