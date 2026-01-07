# SCAT - Spot Classification and Analysis Tool

ML-based analysis tool for *Drosophila* excreta to classify ROD (Reproductive Oblong Deposits) vs Normal deposits.

📖 **[Detailed Workflow Guide](WORKFLOW.md)** - Step-by-step documentation for each stage

## Installation

```bash
cd scat
pip install -e .

# Or with uv
uv add numpy opencv-python Pillow pandas scikit-learn scipy PySide6 matplotlib seaborn jinja2
uv pip install -e .

# For CNN training (optional):
pip install torch torchvision
```

## Quick Start - GUI

```bash
# Launch the main GUI application
uv run python -m scat.cli gui
```

The GUI provides:
- **Labeling**: Annotate deposits for training
- **Training**: Train Random Forest or CNN models
- **Analysis**: Run analysis with progress tracking
- **Results**: View statistics, visualizations, and export reports

## Command Line Usage

### Labeling
```bash
uv run python -m scat.cli label
```

### Training
```bash
# Random Forest classifier
uv run python -m scat.cli train --image-dir ./images --output model_rf.pkl --model-type rf

# U-Net segmentation (requires PyTorch)
# Use GUI: Training tab → Model Type: U-Net Segmentation
```

### Analysis
```bash
# Full analysis with all features
uv run python -m scat.cli analyze ./images --model-type rf --model-path model_rf.pkl -o results --annotate --visualize --stats
```

## Output Structure

```
results/
├── film_summary.csv           # Per-film statistics
├── all_deposits.csv           # Combined deposit data
├── deposits/                  # Individual CSV per image
├── annotated/                 # Annotated images
├── visualizations/            # Plots (PCA, Heatmap, Density maps, etc.)
│   ├── dashboard.png
│   ├── pca_plot.png
│   ├── density_map.png
│   ├── nnd_histogram.png
│   └── ...
├── report.html                # Comprehensive HTML report
└── statistics_report.txt      # Statistical tests
```

## Features

### Detection
- **Rule-based**: Two-stage adaptive thresholding (standard + sensitive)
- **U-Net**: Pixel-level segmentation with learned detection (requires training)

### Classification
- **Threshold**: Circularity-based (no training needed)
- **Random Forest**: 7-feature ML model (fast, accurate)
- **CNN**: Deep learning with image patches

### Visualizations
- PCA plot
- Feature heatmap
- Violin/Box plots
- Density maps (spatial)
- NND histogram
- Clark-Evans clustering summary

### Statistical Analysis
- Normality tests (Shapiro-Wilk)
- Group comparisons (t-test / Mann-Whitney U)
- Multiple comparison correction (Holm-Bonferroni)
- Effect size (Cohen's d)

### Spatial Analysis
- Nearest Neighbor Distance (NND)
- Clark-Evans clustering index
- Quadrant distribution
- Edge vs Center preference
- Density heatmaps

### Reporting
- Automatic HTML report generation
- Excel export
- Summary statistics

## ROD Classification Criteria

- **Circularity** < 0.6 (elongated shape)
- **Lightness** < 0.80 (concentrated, not dilute)

## Python API

```python
from scat import Analyzer, ClassifierConfig, SpatialAnalyzer
from scat.report import generate_report

# Analyze
config = ClassifierConfig(model_type="rf", model_path="model.pkl")
analyzer = Analyzer(classifier_config=config)
result = analyzer.analyze_image("image.tif")

# Spatial analysis
spatial = SpatialAnalyzer()
spatial_result = spatial.analyze(result.deposits, image.shape[:2])
print(f"Clark-Evans R: {spatial_result.clark_evans_r:.2f}")

# Generate report
generate_report(film_summary, output_dir="./results", format="html")
```
