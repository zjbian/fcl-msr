# fCLI-MSR: Play No Favorites — Matching and Balancing Multimodal Sequential Representation for Recommendation

This repository contains the official implementation of **fCLI-MSR**, built upon [CIKM2020-S3Rec](https://github.com/aHuiWang/CIKM2020-S3Rec).

## Overview

We propose a **multi-modal sequential recommendation** framework that integrates item ID, visual (CLIP image), textual (CLIP text), and attribute modalities. The key components are:

- **Multi-gate Mixture of Experts (MMOE)** for adaptive modality fusion
- **Frank-Wolfe solver** for automatic multi-task gradient balancing
- **CLIP-pretrained features** as multi-modal item representations

Supported fusion architectures: `MMOE`, `MLP`, `Transformer`.

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── run.sh                       # Unified experiment script (all-in-one)
│
├── 🔑 Core Contribution
│   ├── dataset.py               # Dataset classes (SASRecDataset, OODDataset)
│   ├── model.py                 # S3RecModel with MMOE/MLP/Transformer fusion + Frank-Wolfe
│   ├── trainer.py               # Multi-task trainer with auto-weight & loss plotting
│   └── run_test.py              # Main entry point (training & evaluation)
│
├── 🧱 Shared Modules
│   ├── modules.py               # Transformer building blocks (Encoder, Attention, etc.)
│   └── utils.py                 # Utilities (metrics, data loading, early stopping)
│
├── 📊 Baseline Comparison (optional)
│   ├── dataset_baseline.py      # Original S3Rec dataset
│   ├── model_baseline.py        # S3Rec, GRU4Rec, SRGNN baselines
│   ├── trainer_baseline.py      # Original S3Rec trainer
│   ├── run_experiment.py        # Table 2: baseline model comparison
│   └── run_introduction_experiment.py  # Loss function comparison
│

├── data/                        # 📦 Four benchmark datasets
│   ├── Scientific/
│   ├── Pantry/
│   ├── Arts/
│   └── Instruments/
│
└── reproduce_ours/              # 📦 Pretrained model checkpoints
    ├── Scientific-epochs-150.pt
    ├── Pantry-epochs-200.pt
    ├── Arts-epochs-200.pt
    └── Instruments-epochs-200.pt
```

## Datasets

We use four Amazon review datasets with multi-modal features:

| Dataset | Users | Items | Interactions | Modality |
|---------|-------|-------|-------------|----------|
| Scientific | 13,100 | 4,386 | 250k | Image + Text (CLIP) |
| Pantry | 6,065 | 1,714 | 128k | Image + Text (CLIP) |
| Arts | 45,486 | 7,791 | 577k | Image + Text (CLIP) |
| Instruments | 24,962 | 5,771 | 302k | Image + Text (CLIP) |

Data format: each line is `user_id item1 item2 item3 ...`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
# Train with MMOE fusion on Scientific dataset
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE --ckp 150

# Train with MLP fusion
python run_test.py --output_dir output/ --data_name Scientific --Ours --MLP --ckp 150

# Train with Transformer fusion
python run_test.py --output_dir output/ --data_name Scientific --Ours --Trans --ckp 150

# Enable automatic multi-task weighting (Frank-Wolfe)
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE --auto_weight
```

### 3. Evaluate

```bash
# Evaluate a trained model
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE --do_eval

# Evaluate with embedding visualization
python run_test.py --output_dir output/ --data_name Scientific --Ours --MMOE --do_eval --plot_eval
```

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
| `--MMOE` | - | MMOE fusion architecture |
| `--MLP` | - | MLP gating fusion |
| `--Trans` | - | Transformer fusion |
| `--auto_weight` | - | Auto multi-task weighting (Frank-Wolfe) |
| `--ckp` | 200 | Pretraining epochs (150 for Scientific) |
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
| 4 | Remove MMOE (single expert) |
| 5 | Remove text modality |
| 6 | Remove image modality |

## Note

- The pretrained checkpoints in `reproduce_ours/` are required for training (they provide S3Rec-pretrained weights)
- If the checkpoint is not found, the model falls back to SASRec (random initialization)
- The data path in `run_test.py` (`--data_dir`) may need to be adjusted for your environment
- For baseline comparison (Table 2), use `run_experiment.py` which supports GRU4Rec, SRGNN, and SASRec

## Citation

If you find this work useful, please cite our paper.

## License

This project is built upon [CIKM2020-S3Rec](https://github.com/aHuiWang/CIKM2020-S3Rec). Please refer to their repository for the original license.
