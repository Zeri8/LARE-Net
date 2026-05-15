# LARE-Net

**Lesion-Aware Reliable Experts for Brain Tumor Segmentation with Arbitrary Missing MRI Modalities**

LARE-Net is a research codebase for incomplete multi-modal brain tumor segmentation. The core idea is to explicitly estimate which available MRI modality should be trusted for each tumor subregion under arbitrary missing-modality settings.

## Research Goal

Given up to four MRI modalities:

- T1
- T1ce
- T2
- FLAIR

this project aims to segment BraTS tumor regions under any non-empty modality combination:

- WT: Whole Tumor
- TC: Tumor Core
- ET: Enhancing Tumor

## Planned Pipeline

1. Build a clean HeMIS-SegResNet missing-modality baseline.
2. Replace fixed mean/variance fusion with lesion-aware reliability routing.
3. Add reliable expert banks for modality, lesion, boundary, and uncertainty modeling.
4. Add boundary distance learning to improve HD95.
5. Add uncertainty calibration and optional subset distillation / reliability-guided context modeling.

## Project Structure

```text
LARE-Net/
├── configs/                 # YAML experiment configs
├── data/                    # dataset, transforms, modality masks, label conversion
├── models/                  # SegResNet, HeMIS baseline, LARE-Net modules
├── losses/                  # segmentation, reliability, boundary, uncertainty losses
├── metrics/                 # Dice, HD95, calibration metrics
├── engine/                  # training, validation, inference loops
├── tools/                   # command-line entry points
├── utils/                   # seed, logging, checkpoints
├── docs/                    # notes and experiment documentation
├── scripts/                 # helper scripts
└── external_baselines/      # notes or wrappers for external baseline methods
```

## Current Status

This repository is being initialized. The first milestone is to implement:

- `data/modality_mask.py`
- `data/label_convert.py`
- `models/hemis_segresnet.py`
- `losses/dice_ce.py`
- `tools/train.py`
- `tools/test_15_modalities.py`

## License

To be decided.
