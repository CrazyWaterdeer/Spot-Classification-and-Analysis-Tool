# SCAT Workflow Guide

This document provides detailed explanations of each stage in SCAT.

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐    ┌──────────────┐
│  Labeling   │ →  │   Training   │ →  │  Detection  │ →  │ Classification │ →  │   Analysis   │
│  (Manual)   │    │  (Learning)  │    │  (Finding)  │    │   (Sorting)    │    │  (Results)   │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────────┘    └──────────────┘
```

---

## 1. Labeling

### Purpose
Generate training data by manually specifying deposit locations and types.

### How to Run
```bash
uv run python -m scat.labeling_gui
# Or use the Labeling tab in the main GUI
```

### Features

| Feature | Shortcut | Description |
|---------|----------|-------------|
| Auto-detect | Re-detect | Initial detection using Adaptive Threshold |
| Manual add | A + drag | Draw region → extract contour via Otsu threshold |
| Set label | 1, 2, 3 | Normal, ROD, Artifact |
| Delete | D, Delete | Delete selected deposit(s) |
| Merge | M | Multiple deposits → single contour (for large deposits) |
| Group | G | Multiple deposits → logical group (for fragmented ROD) |
| Ungroup | U | Remove from group |
| Undo | Ctrl+Z | Undo last action (up to 5 times) |
| Save | Ctrl+S | Save labels |

### Merge vs Group

| Feature | Merge | Group |
|---------|-------|-------|
| Use case | Large deposit partially detected | Fragmented ROD pieces |
| Result | Single contour (may include background) | Individual contours preserved |
| Count | 1 deposit | 1 group (multiple segments) |

### Output File Format
```json
// image_name.labels.json
{
  "image_file": "path/to/image.tif",
  "next_group_id": 3,
  "deposits": [
    {
      "id": 0,
      "contour": [[x1,y1], [x2,y2], ...],
      "label": "normal",
      "area": 150.5,
      "circularity": 0.85,
      "x": 100,
      "y": 200,
      "merged": false,
      "group_id": null
    },
    {
      "id": 1,
      "label": "rod",
      "merged": false,
      "group_id": 1
    },
    {
      "id": 2,
      "label": "rod",
      "merged": false,
      "group_id": 1
    }
  ]
}
```

---

## 2. Training

### 2-1. Detection Model (U-Net Segmentation)

Learns to detect deposit regions at pixel level.

| Item | Description |
|------|-------------|
| **Purpose** | Pixel-level deposit detection |
| **Input** | Images + Label JSON files |
| **Training targets** | Normal and ROD only (Artifacts excluded) |
| **Model** | U-Net (Encoder-Decoder with skip connections) |
| **Loss** | BCE (0.5) + Dice Loss (0.5) |
| **Augmentation** | Flip, Rotation, Brightness, Contrast, Noise (13x augmentation) |
| **Optimization** | Adam + ReduceLROnPlateau + Early Stopping |
| **Output** | `model_unet.pt` (~50MB) |

#### U-Net Architecture
```
Input (256x256x3)
    ↓
[Encoder] 64 → 128 → 256 → 512
    ↓
[Bottleneck] 1024
    ↓
[Decoder] 512 → 256 → 128 → 64
    ↓
Output (256x256x1) → Sigmoid → Binary Mask
```

#### Data Augmentation (13x)
- Original image
- Horizontal flip
- Vertical flip
- 90°, 180°, 270° rotation
- Brightness ×0.8, ×1.2
- Contrast ×0.8, ×1.2
- Gaussian noise
- Flip + brightness combination

#### Early Stopping
- Patience: 10 epochs (stops if no improvement)
- LR Scheduler: After 5 epochs plateau → LR ×0.5

### 2-2. Classification Models

#### Random Forest
| Item | Description |
|------|-------------|
| **Input** | Color/shape features (7 features) |
| **Features** | mean_hue, mean_saturation, mean_lightness, mean_r, mean_g, mean_b, circularity |
| **Output** | `model_rf.pkl` (~1MB) |

#### CNN
| Item | Description |
|------|-------------|
| **Input** | Image patch (64×64) |
| **Architecture** | Conv → Pool → Conv → Pool → FC |
| **Output** | `model_cnn.pt` (~5MB) |

### Recommended Training Data

| Model | Minimum | Recommended |
|-------|---------|-------------|
| U-Net | 5 images | 20-30 images |
| Random Forest | 50 deposits | 200+ deposits |
| CNN | 100 deposits | 500+ deposits |

---

## 3. Detection

### 3-1. Rule-based (Default, No Training Required)

Two-stage detection captures both normal and dilute deposits.

```
Image
  ↓
Convert to Grayscale
  ↓
┌──────────────────────────────────────┐
│ Stage 1: Standard Detection          │
│ - Adaptive Threshold (block=51, c=10)│
│ - Morphological cleanup (3×3)        │
│ → Normal deposits (precise boundary) │
└──────────────────────────────────────┘
  ↓
┌──────────────────────────────────────┐
│ Stage 2: Sensitive Detection         │
│ - Exclusion mask (Stage 1 + 15px)    │
│ - Lower threshold (c=5)              │
│ - HSV-based detection                │
│ → Additional dilute deposits only    │
└──────────────────────────────────────┘
  ↓
Extract contours (min_area=20, max_area=10000)
  ↓
Deposit List
```

### 3-2. U-Net (Learning-based)

```
Image
  ↓
Resize (256×256) + Normalize (/255)
  ↓
U-Net Inference (GPU/CPU)
  ↓
Sigmoid → Binary mask (threshold=0.5)
  ↓
Resize to original dimensions
  ↓
Extract contours (min_area filter)
  ↓
Deposit List
```

### Comparison

| Item | Rule-based | U-Net |
|------|-----------|-------|
| Training required | ❌ | ✅ |
| Artifact detection | High (many) | **Low (few)** |
| Dilute deposits | Two-stage process | **Learned** |
| Boundary accuracy | Moderate | **High** (pixel-level) |
| Speed (CPU) | Fast | Moderate |
| Speed (GPU) | - | **Fast** |
| Parameter tuning | Required | Not needed |

---

## 4. Classification

### 4-1. Threshold (Rule-based)

```python
if circularity < threshold:  # default: 0.6
    label = "rod"
else:
    label = "normal"
```

- ✅ No training required
- ❌ Cannot classify Artifacts
- ❌ Low accuracy

### 4-2. Random Forest

```
Deposit
  ↓
Feature Extraction (FeatureExtractor)
├── mean_hue (0-180)
├── mean_saturation (0-1)
├── mean_lightness (0-1)
├── mean_r, mean_g, mean_b (0-255)
└── circularity (0-1)
  ↓
Random Forest (100 trees, default)
  ↓
Label (normal/rod/artifact) + Confidence
```

- ✅ Fast
- ✅ High accuracy
- ✅ Can classify Artifacts

### 4-3. CNN

```
Deposit
  ↓
Extract image patch (64×64, padding=5)
  ↓
Normalize
  ↓
CNN Forward Pass
  ↓
Softmax
  ↓
Label + Confidence
```

- ✅ Learns image patterns directly
- ❌ Requires more training data
- ❌ Slower (GPU recommended)

### Comparison

| Item | Threshold | Random Forest | CNN |
|------|-----------|--------------|-----|
| Training required | ❌ | ✅ | ✅ |
| Accuracy | Low | **High** | High |
| Speed | Fast | Fast | Slow |
| Artifact classification | ❌ | ✅ | ✅ |
| Data requirement | - | Low | High |

---

## 5. Analysis

### Pipeline

```
Image Folder + (optional) Groups CSV + (optional) Model Files
  ↓
┌────────────────────────────────────────┐
│ For each image:                        │
│   1. Load image                        │
│   2. Detection (Rule-based or U-Net)   │
│   3. Feature extraction (7 features)   │
│   4. Classification (RF/CNN/Threshold) │
│   5. Save individual results           │
└────────────────────────────────────────┘
  ↓
Aggregate results
  ↓
Statistical analysis (if groups provided)
  ↓
Generate visualizations
  ↓
Output reports
```

### Output Files

```
results/
├── film_summary.csv          # Per-image summary
│   └── filename, n_normal, n_rod, n_artifact, n_total, rod_fraction, group
├── deposit_data.csv          # Individual deposit data
│   └── filename, id, x, y, area, circularity, hue, label, confidence
├── statistics_report.txt     # Statistical analysis results
├── annotated/                # Detection visualization
│   └── image_annotated.png
├── deposits/                 # Contour data (for editing)
│   └── image_deposits.json
└── visualizations/           # Charts and graphs
    ├── dashboard.png
    ├── pca_plot.png
    ├── violin_plot.png
    └── ...
```

### Statistical Analysis

```
Groups CSV (group column)
  ↓
Separate data by group
  ↓
┌──────────────────────────────┐
│ Normality test (Shapiro-Wilk)│
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ 2 groups: t-test             │
│           (or Mann-Whitney)  │
│ 3+ groups: ANOVA             │
│           (or Kruskal-Wallis)│
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ Pairwise comparisons         │
│ + Bonferroni correction      │
└──────────────────────────────┘
  ↓
p-value, Effect size (Cohen's d), Significance determination
```

---

## Recommended Settings

### Best Accuracy (Training Required)

| Stage | Method | Model File |
|-------|--------|------------|
| Detection | **U-Net** | model_unet.pt |
| Classification | **Random Forest** | model_rf.pkl |

### Quick Start (No Training)

| Stage | Method |
|-------|--------|
| Detection | Rule-based (sensitive mode) |
| Classification | Threshold (circularity < 0.6) |

---

## Keyboard Shortcuts

### Labeling GUI

| Shortcut | Function |
|----------|----------|
| 1 | Label as Normal |
| 2 | Label as ROD |
| 3 | Label as Artifact |
| D / Delete | Delete selected |
| M | Merge selected |
| G | Group selected |
| U | Ungroup selected |
| S | Select mode |
| A | Add mode |
| N | Next deposit |
| P | Previous deposit |
| Ctrl+S | Save |
| Ctrl+Z | Undo |

### Edit Deposits (Results Tab)

| Shortcut | Function |
|----------|----------|
| 1 | Label as Normal |
| 2 | Label as ROD |
| 3 | Label as Artifact |
| D / Delete | Delete selected |
| Ctrl+Z | Undo |
| Esc | Close dialog |
