"""Training entry point for HeMIS-SegResNet baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.brats_dataset import BraTSDataset
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from losses.dice_ce import DiceBCELoss
from models.hemis_segresnet import HeMISSegResNet
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg.get("name", "hemis_segresnet")
    if name != "hemis_segresnet":
        raise ValueError(f"Only hemis_segresnet is supported for baseline training, got {name}.")
    return HeMISSegResNet(
        num_modalities=model_cfg.get("num_modalities", 4),
        in_channels_per_modality=model_cfg.get("in_channels_per_modality", 1),
        num_classes=model_cfg.get("num_classes", 3),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128, 256])),
        norm=model_cfg.get("norm", "instance"),
        act=model_cfg.get("activation", "leaky_relu"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeMIS-SegResNet baseline.")
    parser.add_argument("--config", type=str, default="configs/brats_lare.yaml")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root in config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true", help="Use small model and short run for smoke testing.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))

    data_cfg = config["data"]
    train_cfg = config["training"]
    log_cfg = config["logging"]

    if args.data_root is not None:
        data_cfg["root"] = args.data_root
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers

    if args.debug:
        config["model"]["encoder_channels"] = [8, 16, 32, 64]
        train_cfg["epochs"] = min(int(train_cfg.get("epochs", 2)), 2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    dataset = BraTSDataset(
        root=data_cfg["root"],
        label_mode=data_cfg.get("label_mode", "brats_1_2_4"),
        normalize=True,
        crop_size=tuple(data_cfg.get("patch_size", [128, 128, 128])),
        random_crop=True,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset is too small for the requested validation split.")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    model = build_model(config).to(device)
    criterion = DiceBCELoss(
        dice_weight=float(config["loss"]["segmentation"].get("dice_weight", 1.0)),
        bce_weight=float(config["loss"]["segmentation"].get("bce_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    save_dir = Path(log_cfg.get("save_dir", "checkpoints/brats_lare"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    epochs = int(train_cfg.get("epochs", 300))
    val_interval = int(train_cfg.get("val_interval", 5))
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=bool(train_cfg.get("amp", True)),
            grad_clip=float(train_cfg.get("grad_clip", 0.0)),
            min_modalities=int(config["missing_modality"].get("min_modalities", 1)),
            max_modalities=int(config["missing_modality"].get("max_modalities", 4)),
        )
        print(f"Epoch {epoch}: train {train_metrics}")

        if epoch % val_interval == 0 or epoch == epochs:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                amp=bool(train_cfg.get("amp", True)),
                fixed_mask=[1, 1, 1, 1],
                compute_hd95=False,
            )
            print(f"Epoch {epoch}: val {val_metrics}")
            current_dice = val_metrics.get("dice_Avg", 0.0)
            save_checkpoint(save_dir / "last.pt", model, optimizer, epoch=epoch, best_metric=best_dice, extra={"config": config})
            if current_dice > best_dice:
                best_dice = current_dice
                save_checkpoint(save_dir / "best.pt", model, optimizer, epoch=epoch, best_metric=best_dice, extra={"config": config})
                print(f"New best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
