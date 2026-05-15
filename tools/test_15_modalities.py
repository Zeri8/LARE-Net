"""Evaluate a trained baseline under all 15 non-empty modality combinations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from data.brats_dataset import BraTSDataset
from data.modality_mask import ALL_MODALITY_MASKS, mask_to_name
from engine.validate import validate
from losses.dice_ce import DiceBCELoss
from models.hemis_segresnet import HeMISSegResNet
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    return HeMISSegResNet(
        num_modalities=model_cfg.get("num_modalities", 4),
        in_channels_per_modality=model_cfg.get("in_channels_per_modality", 1),
        num_classes=model_cfg.get("num_classes", 3),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128, 256])),
        norm=model_cfg.get("norm", "instance"),
        act=model_cfg.get("activation", "leaky_relu"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test all 15 modality combinations.")
    parser.add_argument("--config", type=str, default="configs/brats_lare.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compute-hd95", action="store_true")
    parser.add_argument("--output-csv", type=str, default="outputs/test_15_modalities.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    if args.data_root is not None:
        config["data"]["root"] = args.data_root

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    dataset = BraTSDataset(
        root=config["data"]["root"],
        label_mode=config["data"].get("label_mode", "brats_1_2_4"),
        normalize=True,
        crop_size=tuple(config["data"].get("patch_size", [128, 128, 128])),
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    criterion = DiceBCELoss(
        dice_weight=float(config["loss"]["segmentation"].get("dice_weight", 1.0)),
        bce_weight=float(config["loss"]["segmentation"].get("bce_weight", 1.0)),
    )

    rows: List[Dict[str, float | str]] = []
    for mask in ALL_MODALITY_MASKS:
        name = mask_to_name(mask)
        print(f"Evaluating mask {mask}: {name}")
        metrics = validate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            amp=bool(config["training"].get("amp", True)),
            fixed_mask=mask,
            threshold=float(config["inference"].get("threshold", 0.5)),
            compute_hd95=args.compute_hd95,
        )
        row: Dict[str, float | str] = {"mask": str(mask), "modalities": name}
        row.update(metrics)
        rows.append(row)
        print(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["mask", "modalities"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    main()
