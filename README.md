# fCLI-MSR: Play No Favorites — Matching and Balancing Multimodal Sequential Representation for Recommendation

This repository contains the official implementation of **fCLI-MSR**, built upon [CIKM2020-S3Rec](https://github.com/aHuiWang/CIKM2020-S3Rec).

## Overview

We propose a **multi-modal sequential recommendation** framework that integrates item ID, visual (CLIP image), textual (CLIP text), and attribute modalities. The key components are:

- **Multi-modal Expert Network (MEN)** for adaptive modality fusion
- **Frank-Wolfe solver** for automatic multi-task gradient balancing
- **CLIP-pretrained features** as multi-modal item representations

Supported fusion architectures: `MEN` (MMOE), `MLP`, `Transformer`.

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── run.sh                       # Unified experiment script (all-in-one)
│
├── 🔑 Core Contribution
│   ├── dataset.py               # Dataset classes (SASRecDataset, OODDataset)
│   ├── model.py                 # S3RecModel + MEN(MMOE)/MLP/Transformer fusion + Frank-Wolfe solver
│   ├── trainer.py               # Multi-task trainer with auto-weight & loss plotting
│   └── run_test.py              # Main entry point (training & evaluation)
│
├── 🧱 Shared Modules
│   ├── modules.py               # Transformer building blocks (Encoder, Attention, etc.)
│   └── utils.py                 # Utilities (metrics, data loading, early stopping)
│
├── 📊 Baseline Comparison (optional)
│   └── baseline/
│       ├── dataset.py           # Original S3Rec dataset
│       ├── model.py             # S3Rec, GRU4Rec, SRGNN baselines
│       ├── trainer.py           # Original S3Rec trainer
│       ├── run_experiment.py    # Table 2: baseline model comparison
│       └── run_introduction.py  # Loss function comparison
│
└── data/                        # 📦 Four benchmark datasets
    ├── Scientific/
    ├── Pantry/
    ├── Arts/
    └── Instruments/
```

## Preparation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare CLIP features (required)

This codebase uses **pre-extracted CLIP features** for image and text modalities. These are pre-computed offline using a fine-tuned CLIP model (ViT-B/16 + BERT) and stored as `.pkl` files. They are **not included** in this repository due to size.

For each dataset, you need two `.pkl` files containing a dict with keys `img_weights` and `text_weights`:
- `<DatasetName>_clip_weights_finetune_minloss.pkl` — CLIP features fine-tuned to minimize matching loss
- Shape: `img_weights` and `text_weights` are both `(num_items, clip_hidden_dim)` arrays

After preparing the CLIP `.pkl` files, update the path in `run_test.py` (line ~161) to point to your files.

### 3. Prepare datasets

Download the Amazon review datasets and place under `data/<DatasetName>/`:
- `<DatasetName>.txt` — user interaction sequences (format: `user_id item1 item2 ...`)
- `<DatasetName>_item2attributes.json` — item-to-attribute mapping

The four preprocessed datasets used in the paper are from the [Amazon Product Reviews](https://nijianmo.github.io/amazon/index.html) collection (Scientific, Pantry, Instruments, Arts).

## Training

**Training from scratch is fully supported** — no pretrained checkpoints required. The model initializes with random weights and trains end-to-end.

```bash
# Train with MEN fusion on Scientific dataset (from scratch)
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE

# Train with MLP fusion
python run_test.py --output_dir output/ --data_name Scientific --Ours --MLP

# Train with Transformer fusion
python run_test.py --output_dir output/ --data_name Scientific --Ours --Trans

# Enable automatic multi-task weighting (Frank-Wolfe solver)
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE --auto_weight

### 4. Run ablation / hyperparameter search

```bash
# All experiments via unified script
bash run.sh ablation  DATASET=Scientific   # Ablation study
bash run.sh lambda    DATASET=Scientific   # λ1/λ2 grid search
bash run.sh mmoe      DATASET=Scientific   # Expert count search
bash run.sh baseline                       # Table 2 baselines
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_name` | Scientific | Dataset: Scientific, Pantry, Arts, Instruments |
| `--Ours` | - | Enable multi-modal mode (required) |
| `--MMOE` | - | MEN fusion architecture |
| `--MLP` | - | MLP gating fusion |
| `--Trans` | - | Transformer fusion |
| `--auto_weight` | - | Auto multi-task weighting (Frank-Wolfe solver) |
| `--lambda1` | 1.0 | Image modality weight |
| `--lambda2` | 3.0 | Text modality weight |
| `--main_expert_num` | 4 | Number of shared experts |
| `--modal_expert_num` | 10 | Number of modality-specific experts |
| `--lr` | 0.002 | Learning rate |
| `--batch_size` | 256 | Batch size |
| `--epochs` | 150 | Training epochs |
| `--patience` | 15 | Early stopping patience |
| `--do_eval` | - | Evaluation only (skip training) |
| `--gpu_id` | 0 | GPU device ID |
| `--no_cuda` | - | Force CPU mode |

## Ablation Codes

Use `--ablation_code` to study individual components:

| Code | Component |
|------|-----------|
| 1 | Full model (baseline) |
| 2 | Remove CLIP feature extraction |
| 3 | Remove attribute encoder |
| 4 | Remove MEN (single expert) |
| 5 | Remove text modality |
| 6 | Remove image modality |

## Note

- **Training from scratch is fully supported** — no pretrained checkpoints needed
- CLIP features must be pre-extracted and provided as `.pkl` files (see [Preparation](#2-prepare-clip-features-required))
- The data path in `run_test.py` (`--data_dir`) may need adjustment for your environment
- For baseline comparison (Table 2), use `baseline/run_experiment.py`

## Reproducibility

All random seeds are fixed via `utils.set_seed()` which sets seeds for Python `random`, NumPy, PyTorch, and CUDA, along with `cudnn.deterministic=True` and `cudnn.benchmark=False`. The seed is configurable via `--seed` (default: 23).

## Citation

If you find this work useful, please cite our paper.

## License

This project is built upon [CIKM2020-S3Rec](https://github.com/aHuiWang/CIKM2020-S3Rec). Please refer to their repository for the original license.
