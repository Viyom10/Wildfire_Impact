# 🔥 Wildfire Impact Detection - Complete Project Documentation

> **Multi-Hazard Environmental Detection System using Satellite Imagery and Deep Learning**

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Start](#quick-start)
3. [Project Overview](#project-overview)
4. [Technical Architecture](#technical-architecture)
5. [Installation Guide](#installation-guide)
6. [Usage Guide](#usage-guide)
7. [Model Details](#model-details)
8. [Data Pipeline](#data-pipeline)
9. [Configuration Reference](#configuration-reference)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Performance Metrics](#performance-metrics)

---

## 🎯 Executive Summary

### What is this project?

**Wildfire Impact Detection** is an AI-powered system that automatically detects environmental damage from satellite images. It can identify:

- **🔥 Wildfire burnt areas** - Regions damaged by forest fires
- **🏜️ Drought-affected zones** - Areas showing vegetation stress

### Who is it for?

| Audience | Use Case |
|----------|----------|
| **Environmental Agencies** | Rapid disaster assessment |
| **Researchers** | Climate change impact studies |
| **Emergency Services** | Resource allocation planning |
| **Students/Developers** | Learning deep learning for remote sensing |
| **Data Scientists** | Benchmarking change detection models |

### Key Features

✅ **10 Pre-built Neural Networks** - From simple U-Net to state-of-the-art transformers  
✅ **Multi-Hazard Support** - Switch between wildfire and drought detection  
✅ **Multiple Input Formats** - GeoTIFF, NumPy arrays, JPG/PNG images  
✅ **Pre-trained Model** - BAM-CD model ready for immediate use  
✅ **Web Interface** - Drag-and-drop browser-based prediction  
✅ **Synthetic Data Generator** - Test without downloading real datasets  

---

## ⚡ Quick Start

### 30-Second Demo

```bash
# 1. Setup
cd Wildfire_Impact
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test data
python3 create_synthetic_dataset.py --event_type wildfire

# 3. Run prediction
python3 predict_from_images.py --demo
```

### Predict from Your Own Images

```bash
# Wildfire detection
python3 predict_from_images.py --before image_before.jpg --after image_after.jpg --rgb --output result.png

# Drought detection
python3 predict_from_images.py --before image_before.jpg --after image_after.jpg --rgb --drought --output result.png
```

---

## 🌍 Project Overview

### The Problem

When a wildfire occurs, emergency responders need to know:
- **Where** exactly did the fire burn?
- **How much** area was affected?
- **What** type of land was damaged?

Manual analysis of satellite images is slow (days) and expensive. This project provides automated analysis in **minutes**.

### The Solution

```
┌─────────────────┐     ┌─────────────────┐
│ Before Image    │     │ After Image     │
│ (Pre-event)     │     │ (Post-event)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────────┐
          │   Neural Network    │
          │   (BAM-CD Model)    │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │   Prediction Map    │
          │  0 = Unchanged      │
          │  1 = Affected       │
          └─────────────────────┘
```

### How It Works (Non-Technical)

Think of it like "spot the difference" between two photos:

1. **Before Photo**: Satellite image of an area before a fire
2. **After Photo**: Same area after the fire
3. **AI Analysis**: The neural network compares both images pixel-by-pixel
4. **Result**: A colored map showing exactly which areas burned

The AI has been trained on hundreds of real wildfire events, so it knows what burnt vegetation "looks like" in satellite imagery.

---

## 🏗️ Technical Architecture

### System Components

```
Wildfire_Impact/
├── 🧠 models/                 # 10 Neural Network Architectures
│   ├── bam_cd/               # BAM-CD (Best performing)
│   ├── unet.py               # Classic U-Net
│   ├── changeformer.py       # Transformer-based
│   ├── snunet.py             # Dense connections
│   ├── fc_ef_diff.py         # Fast & lightweight
│   └── ...                   # 5 more architectures
│
├── ⚙️ configs/                # JSON Configuration Files
│   ├── config.json           # Training configuration
│   ├── config_eval*.json     # Evaluation configurations
│   └── method/*.json         # Model-specific settings
│
├── 📊 data/                   # Data Storage
│   └── processed/            # Preprocessed patches
│
├── 📜 Core Scripts
│   ├── run_experiment.py     # Training & evaluation
│   ├── predict_from_images.py # Inference on new images
│   ├── create_dataset.py     # Data preprocessing
│   ├── create_synthetic_dataset.py # Test data generator
│   ├── visualize_predictions.py # Result visualization
│   └── web_interface.py      # Browser-based UI
│
├── 🔧 Utilities
│   ├── utils.py              # Helper functions
│   ├── dataset_utils.py      # Data loading
│   └── cd_experiments_utils.py # Training loops
│
└── 📁 losses/                 # Loss Functions
    ├── dice.py               # Dice loss
    └── bce_and_dice.py       # Combined loss
```

### Neural Network Architecture

The project uses a **Siamese (twin) architecture**:

```
Before Image ──┐
               ├──► Shared Encoder ──┐
After Image  ──┘                     │
                                     ├──► Feature Fusion ──► Decoder ──► Prediction
Before Image ──┐                     │
               ├──► Shared Encoder ──┘
After Image  ──┘
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| **Encoder** | Extracts features from images (ResNet backbone) |
| **Fusion** | Combines before/after features (concatenation or difference) |
| **Decoder** | Upsamples features to pixel-level predictions |
| **Segmentation Head** | Final classification (changed/unchanged) |

### Data Flow Pipeline

```
Raw Satellite Data (HDF5)
         │
         ▼ create_dataset.py
Preprocessed Patches (.npy)
         │
         ▼ dataset_utils.py
PyTorch DataLoader
         │
         ▼ run_experiment.py
Trained Model (.pt)
         │
         ▼ predict_from_images.py
Prediction Maps
```

---

## 💻 Installation Guide

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Optional | NVIDIA with 8GB+ VRAM |
| **Storage** | 2 GB | 10 GB (with datasets) |

### Step-by-Step Installation

#### macOS / Linux

```bash
# 1. Clone or download the project
cd ~/Downloads/Wildfire_Impact

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### Windows

```cmd
:: 1. Navigate to project
cd %USERPROFILE%\Downloads\Wildfire_Impact

:: 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `pyjson5` | JSON config parsing |
| `torchmetrics` | Evaluation metrics |
| `gradio` | Web interface (optional) |
| `rasterio` | GeoTIFF support (optional) |

---

## 📖 Usage Guide

### 1. Predict from Images (Most Common)

#### Command Line

```bash
# Basic wildfire detection (RGB images)
python3 predict_from_images.py \
    --before path/to/before.jpg \
    --after path/to/after.jpg \
    --rgb \
    --output result.png

# Drought detection
python3 predict_from_images.py \
    --before path/to/before.jpg \
    --after path/to/after.jpg \
    --rgb \
    --drought \
    --output drought_result.png

# With custom threshold (more sensitive)
python3 predict_from_images.py \
    --before before.jpg \
    --after after.jpg \
    --rgb \
    --threshold 0.3 \
    --output result.png

# Save binary mask
python3 predict_from_images.py \
    --before before.jpg \
    --after after.jpg \
    --rgb \
    --output-mask mask.npy \
    --output-prob probabilities.npy
```

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--before` | Path to before image | Required |
| `--after` | Path to after image | Required |
| `--rgb` | Input is RGB (3-channel) | False |
| `--drought` | Drought detection mode | False |
| `--output` | Save visualization path | None |
| `--output-mask` | Save binary mask (.npy) | None |
| `--output-prob` | Save probability map (.npy) | None |
| `--threshold` | Detection threshold | 0.5 |
| `--no-show` | Don't display visualization | False |
| `--demo` | Run demo with synthetic data | False |

### 2. Web Interface

```bash
# Install Gradio (if not already)
pip install gradio

# Launch web interface
python3 web_interface.py

# Open browser to http://localhost:7860
```

**Features:**
- Drag-and-drop image upload
- Real-time predictions
- Interactive visualization
- Download results

### 3. Train Custom Model

```bash
# Generate training data
python3 create_synthetic_dataset.py --event_type wildfire

# Train model
python3 run_experiment.py --config configs/config.json
```

### 4. Evaluate Pre-trained Model

```bash
# Wildfire evaluation
python3 run_experiment.py --config configs/config_eval_synthetic.json

# Drought evaluation
python3 run_experiment.py --config configs/config_eval_synthetic_drought.json
```

### 5. Visualize Results

```bash
python3 visualize_predictions.py \
    --results_path results/viz/ \
    --config configs/config_eval_synthetic.json \
    --mode test
```

---

## 🧠 Model Details

### Available Models (10 Total)

| Model | Type | Parameters | Best For | Paper |
|-------|------|------------|----------|-------|
| **BAM-CD** | CNN | ~25M | Overall best performance | - |
| **U-Net** | CNN | ~7M | Simple, reliable baseline | [2015](https://arxiv.org/abs/1505.04597) |
| **FC-EF-Diff** | CNN | ~1M | Fast inference | [2018](https://arxiv.org/abs/1810.08462) |
| **FC-EF-Conc** | CNN | ~1M | Simple fusion | [2018](https://arxiv.org/abs/1810.08462) |
| **SNUNet** | CNN | ~12M | Dense feature connections | [2021](https://arxiv.org/abs/2101.09218) |
| **ChangeFormer** | Transformer | ~40M | Global context | [2022](https://arxiv.org/abs/2201.01293) |
| **BIT-CD** | Transformer | ~30M | Attention-based | [2021](https://arxiv.org/abs/2109.07373) |
| **HFANet** | CNN | ~15M | Hierarchical features | - |
| **ADHR-CDNet** | CNN | ~20M | Attention + dense | - |
| **TransUNet-CD** | Hybrid | ~35M | CNN + Transformer | - |

### Model Selection Guide

```
┌─────────────────────────────────────────────────────────┐
│ Need best accuracy?        → BAM-CD                     │
│ Need fast inference?       → FC-EF-Diff                 │
│ Need simple baseline?      → U-Net                      │
│ Have limited GPU memory?   → FC-EF-Diff or FC-EF-Conc   │
│ Want transformer power?    → ChangeFormer               │
│ Research/experimentation?  → Try multiple models        │
└─────────────────────────────────────────────────────────┘
```

### BAM-CD Architecture (Flagship Model)

```
Input (Before + After images)
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ResNet34│ │ResNet34│  ← Shared weights (Siamese)
│Encoder │ │Encoder │
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         │ (Concatenate features)
         ▼
   ┌───────────┐
   │  UNet     │
   │  Decoder  │  ← Skip connections
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │Segmentation│
   │   Head    │
   └─────┬─────┘
         │
         ▼
   Binary Mask (H×W)
```

---

## 📊 Data Pipeline

### Supported Input Formats

| Format | Extension | Channels | Best For |
|--------|-----------|----------|----------|
| **GeoTIFF** | .tif, .tiff | 9 (Sentinel-2) | Actual satellite data |
| **NumPy** | .npy | Any | Preprocessed data |
| **Standard Images** | .png, .jpg | 3 (RGB) | Quick testing |

### Sentinel-2 Bands Used

| Band | Name | Wavelength | Purpose |
|------|------|------------|---------|
| B02 | Blue | 490 nm | Visible |
| B03 | Green | 560 nm | Visible |
| B04 | Red | 665 nm | Visible |
| B05 | Red Edge 1 | 705 nm | Vegetation |
| B06 | Red Edge 2 | 740 nm | Vegetation |
| B07 | Red Edge 3 | 783 nm | Vegetation |
| B8A | NIR | 865 nm | Vegetation health |
| B11 | SWIR 1 | 1610 nm | Burn detection |
| B12 | SWIR 2 | 2190 nm | Burn detection |

### Creating Synthetic Data

```bash
# Wildfire data
python3 create_synthetic_dataset.py --event_type wildfire

# Drought data
python3 create_synthetic_dataset.py --event_type drought
```

This creates:
- 10 training patches
- 3 validation patches
- 3 test patches
- `.pkl` manifest files

### Data Directory Structure

```
data/processed/
└── sen2_20_mod_500/
    ├── synthetic_train.pkl
    ├── synthetic_val.pkl
    ├── synthetic_test.pkl
    └── 2024/
        ├── synthetic_event_train_000_patch_000.S2_before.npy
        ├── synthetic_event_train_000_patch_000.S2_after.npy
        ├── synthetic_event_train_000_patch_000.label.npy
        └── ...
```

---

## ⚙️ Configuration Reference

### Main Configuration (config.json)

```json
{
    "method": "bam_cd",           // Model architecture
    "mode": "train",              // "train" or "eval"
    "event_type": "wildfire",     // "wildfire" or "drought"
    "dataset_type": "sen2_20_mod_500",
    
    "train": {
        "n_epochs": 150,          // Training epochs
        "batch_size": 8,          // Samples per batch
        "loss_function": "dice+ce", // Loss function
        "mixed_precision": false  // FP16 training
    },
    
    "paths": {
        "dataset": "data/processed/",
        "results": "results/",
        "load_state": null        // Checkpoint path
    },
    
    "datasets": {
        "train": "synthetic_train.pkl",
        "val": "synthetic_val.pkl",
        "test": "synthetic_test.pkl",
        "data_source": "sen2",    // "sen2" or "mod"
        "scale_input": "clamp_scale_10000"
    }
}
```

### Model-Specific Configs

Located in `configs/method/`:

| File | Model |
|------|-------|
| `bam_cd.json` | BAM-CD settings |
| `unet.json` | U-Net settings |
| `changeformer.json` | ChangeFormer settings |
| ... | ... |

### Available Loss Functions

| Loss | Description | Config Value |
|------|-------------|--------------|
| Cross Entropy | Standard classification | `"cross_entropy"` |
| Dice | Overlap-based | `"dice"` |
| Dice + CE | Combined (recommended) | `"dice+ce"` |
| Focal | Class imbalance handling | `"focal"` |

---

## 🔧 API Reference

### predict_from_images.py

```python
# Main prediction function
def predict(model, device, before_img, after_img):
    """
    Run prediction on before/after image pair.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Loaded model
    device : torch.device
        CPU or CUDA device
    before_img : torch.Tensor
        Preprocessed before image (1, C, H, W)
    after_img : torch.Tensor
        Preprocessed after image (1, C, H, W)
    
    Returns:
    --------
    pred_mask : numpy.ndarray
        Binary prediction (H, W), 0=unchanged, 1=affected
    prob_map : numpy.ndarray
        Probability map (H, W), range [0, 1]
    """
```

### utils.py

```python
def init_model(configs, model_configs, checkpoint, num_channels, device):
    """Initialize a model from configuration."""
    
def create_loss(configs, mode, device, class_weights, model_configs):
    """Create loss function based on config."""
    
def compute_class_weights(train_loader, configs):
    """Compute class weights for imbalanced data."""
```

### dataset_utils.py

```python
class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading satellite image patches.
    
    Parameters:
    -----------
    mode : str
        'train', 'val', or 'test'
    configs : dict
        Configuration dictionary
    """
```

---

## 🔍 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| `CUDA out of memory` | GPU memory full | Reduce `batch_size` in config |
| `FileNotFoundError: dataset` | No data generated | Run `create_synthetic_dataset.py` |
| `No module named 'timm'` | Missing package | `pip install timm` |
| `ValueError: channel mismatch` | Wrong input format | Add `--rgb` flag for JPG/PNG |
| `Low accuracy` | Using RGB approximation | Use actual Sentinel-2 data |

### GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
# Edit config: "gpu_ids": []
```

### Memory Optimization

```json
// In config.json
{
    "datasets": {
        "batch_size": 2,      // Reduce from 8
        "num_workers": 0      // Reduce from 4
    },
    "train": {
        "mixed_precision": true  // Enable FP16
    }
}
```

---

## 📈 Performance Metrics

### Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **IoU** | Intersection over Union | TP / (TP + FP + FN) |
| **F1-Score** | Harmonic mean of precision/recall | 2 × (P × R) / (P + R) |
| **Precision** | Correct positive predictions | TP / (TP + FP) |
| **Recall** | Found positives | TP / (TP + FN) |

### Expected Performance (Synthetic Data)

| Model | Mean IoU | Burnt F1 | Inference Time |
|-------|----------|----------|----------------|
| BAM-CD | ~85% | ~90% | ~50ms/image |
| U-Net | ~80% | ~85% | ~30ms/image |
| FC-EF-Diff | ~75% | ~80% | ~15ms/image |

*Note: Performance varies significantly with real data quality.*

### Output Interpretation

```
📊 Results:
   Total pixels:     65,536    # Image size (256×256)
   Burnt pixels:     12,450    # Detected affected pixels
   Burnt area:       19.00%    # Percentage affected
   Avg burn prob:    0.2134    # Average probability
   Max burn prob:    0.9876    # Peak probability
```

---

## 📚 Additional Resources

### Documentation Files

| File | Content |
|------|---------|
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Copy-paste test commands |
| [QUICK_GUIDE.md](QUICK_GUIDE.md) | Quick reference |
| [EXPLANATION.md](EXPLANATION.md) | Detailed code explanation |
| [ML_INTERVIEW_GUIDE.md](ML_INTERVIEW_GUIDE.md) | Interview preparation |
| [OUTPUT_EXPLAINER.md](OUTPUT_EXPLAINER.md) | Understanding outputs |
| [FEATURES.md](FEATURES.md) | Feature overview |

### External Resources

- **FLOGA Dataset**: Original wildfire dataset for Greece
- **Sentinel-2**: ESA's Earth observation satellite
- **PyTorch**: Deep learning framework documentation

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release |
| 1.1 | Jan 2026 | Added drought detection |
| 1.2 | Feb 2026 | Added predict_from_images.py |
| 1.3 | Feb 2026 | Added --drought flag, improved docs |

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Documentation Last Updated: February 10, 2026*
