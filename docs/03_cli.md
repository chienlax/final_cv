# CLI Reference

Complete command-line interface reference for the multimodal recommendation system.

---

## Quick Start

```bash
# 1. Preprocess data
python src/preprocessing/run_preprocessing.py --dataset electronics

# 2. Train a model
python src/main.py --model lattice --dataset electronics

# 3. Evaluate only
python src/main.py --model lattice --dataset electronics --eval-only
```

---

## 1. Preprocessing (`run_preprocessing.py`)

Converts raw 5-core CSV data into model-ready format with features.

### Usage

```bash
python src/preprocessing/run_preprocessing.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | choice | `electronics` | Dataset to process: `electronics`, `beauty`, `clothing` |
| `--seed-users` | int | `10000` | Number of seed users for sampling |
| `--k-core` | int | `5` | Minimum interactions per user/item |
| `--cold-ratio` | float | `0.2` | Fraction of items held as cold (0.0-1.0) |
| `--output-dir` | str | `data/processed` | Base output directory |
| `--data-dir` | str | `data/raw` | Input data directory |
| `--seed` | int | `2024` | Random seed for reproducibility |
| `--skip-features` | flag | - | Skip feature extraction (for debugging) |

### Examples

```bash
# Default: Electronics with 10k seed users
python src/preprocessing/run_preprocessing.py --dataset electronics

# Beauty with custom seed count
python src/preprocessing/run_preprocessing.py --dataset beauty --seed-users 8000

# Custom cold ratio (30% cold items)
python src/preprocessing/run_preprocessing.py --dataset electronics --cold-ratio 0.3

# Debug: Skip slow feature extraction
python src/preprocessing/run_preprocessing.py --dataset electronics --skip-features
```

### Output

Creates `data/processed/{dataset}/` with:
- `train.txt`, `val.txt`, `test_warm.txt`, `test_cold.txt`
- `feat_visual.npy`, `feat_text.npy`
- `maps.json`, `stat.txt`, `config.json`

---

## 2. Training & Evaluation (`main.py`)

Train models and run the three-track evaluation protocol.

### Usage

```bash
python src/main.py --model MODEL [OPTIONS]
```

### Required Arguments

| Argument | Type | Choices | Description |
|----------|------|---------|-------------|
| `--model` | choice | `lattice`, `micro`, `diffmm` | Model architecture to use |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | `electronics` | Dataset name (must be preprocessed) |
| `--data-dir` | str | `data/processed` | Base directory for processed data |
| `--epochs` | int | `100` | Number of training epochs |
| `--batch-size` | int | `1024` | Training batch size |
| `--lr` | float | `1e-3` | Learning rate |
| `--seed` | int | `2024` | Random seed |
| `--eval-only` | flag | - | Skip training, only evaluate |
| `--checkpoint` | str | - | Path to checkpoint for evaluation |
| `--output-dir` | str | `checkpoints` | Output directory for checkpoints |

### Training Examples

```bash
# Train LATTICE on Electronics
python src/main.py --model lattice --dataset electronics

# Train MICRO with custom epochs
python src/main.py --model micro --dataset electronics --epochs 50

# Train DiffMM with custom batch size
python src/main.py --model diffmm --dataset electronics --batch-size 512

# Train with custom learning rate
python src/main.py --model lattice --dataset electronics --lr 5e-4
```

### Evaluation Examples

```bash
# Evaluate saved LATTICE model
python src/main.py --model lattice --dataset electronics --eval-only

# Evaluate from specific checkpoint
python src/main.py --model micro --dataset electronics --eval-only \
    --checkpoint checkpoints/electronics/micro/best.pt
```

### Output

Training creates `checkpoints/{dataset}/{model}/`:
- `best.pt` — Best model checkpoint
- `training_history.json` — Loss curves
- `eval_results.json` — Three-track evaluation results

---

## 3. EDA (`run_eda.py`)

Exploratory data analysis for datasets.

### Usage

```bash
python src/run_eda.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | choice | - | Dataset: `beauty`, `clothing`, `electronics`, `all` |
| `--output-dir` | str | `docs` | Output directory for reports |
| `--download-images` | flag | - | Download sample images for analysis |
| `--image-sample-size` | int | `500` | Number of images to sample |
| `--academic-analysis` | flag | - | Run extended academic analyses |
| `--seed` | int | `42` | Random seed |

### Examples

```bash
# Run EDA on Electronics
python src/run_eda.py --dataset electronics

# Run EDA on all datasets
python src/run_eda.py --dataset all

# Run with academic analysis (slower but more thorough)
python src/run_eda.py --dataset electronics --academic-analysis
```

---

## 4. Environment Variables

Set these before running scripts for optimal performance:

```bash
# Use 6 P-cores on i5-13500 (avoid E-core latency)
$env:OMP_NUM_THREADS = "6"

# Disable CUDA memory caching (reduce VRAM fragmentation)
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
```

---

## 5. Common Workflows

### Full Pipeline (Electronics)

```bash
# Step 1: Preprocess
python src/preprocessing/run_preprocessing.py --dataset electronics

# Step 2: Train all three models
python src/main.py --model lattice --dataset electronics
python src/main.py --model micro --dataset electronics
python src/main.py --model diffmm --dataset electronics

# Step 3: Compare results
cat checkpoints/electronics/*/eval_results.json
```

### Quick Test (Debug Mode)

```bash
# Preprocess without features (fast)
python src/preprocessing/run_preprocessing.py --dataset electronics --skip-features

# Train for 1 epoch
python src/main.py --model lattice --dataset electronics --epochs 1
```

### Reproducing Results

```bash
# Set same seed as paper/friend
python src/preprocessing/run_preprocessing.py --dataset electronics --seed 2024
python src/main.py --model lattice --dataset electronics --seed 2024
```

---

## 6. Troubleshooting

### Out of Memory (GPU)

```bash
# Reduce batch size
python src/main.py --model lattice --dataset electronics --batch-size 512
```

### Preprocessing Too Slow

```bash
# Skip feature extraction for initial testing
python src/preprocessing/run_preprocessing.py --dataset electronics --skip-features
```

### PowerShell Multi-line Commands

In PowerShell, use single-line commands:
```powershell
# WRONG (multi-line with backslash)
python src/main.py `
    --model lattice `
    --dataset electronics

# CORRECT (single line)
python src/main.py --model lattice --dataset electronics
```
