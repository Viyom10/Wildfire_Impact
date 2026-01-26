# 🔥 Wildfire Impact Detection - Quick Guide

> **One-liner**: This project uses AI to automatically detect burnt areas from satellite images, replacing slow manual analysis.

---

## 1. What This Project Does

**Problem**: After wildfires, we need to know exactly what burned. Manual satellite image analysis is slow, expensive, and inconsistent.

**Solution**: Feed the AI two satellite images (before & after fire) → it outputs a map showing which pixels burned.

```
Before Image + After Image → Neural Network → Burnt Area Map
```

**Real Impact**: Uses data from 326 real Greek wildfires (2017-2021) with ground truth from the Hellenic Fire Service.

---

## 2. How It Works (Architecture)

### The Siamese Approach
```
┌─────────────┐     ┌─────────────┐
│ Before Fire │     │ After Fire  │
└──────┬──────┘     └──────┬──────┘
       ↓                   ↓
   [Encoder]           [Encoder]  ← Same network (twins)
       ↓                   ↓
       └───────┬───────────┘
               ↓
         [Compare Features]
               ↓
           [Decoder]
               ↓
      [Pixel-wise Prediction]
      (0 = No fire, 1 = Burnt)
```

### Data Flow
1. **Raw Data** (HDF5) → `create_dataset.py` → **Patches** (.npy files)
2. **Patches** → `run_experiment.py` → **Trained Model** (.pt file)
3. **Model** → `visualize_predictions.py` → **Visual Results**

---

## 3. Project Structure (Key Files)

| File | Purpose |
|------|---------|
| `run_experiment.py` | Main script - trains or evaluates models |
| `create_dataset.py` | Converts raw satellite data to patches |
| `create_synthetic_dataset.py` | Generates fake data for quick testing |
| `visualize_predictions.py` | Creates visual outputs |
| `dataset_utils.py` | Loads and preprocesses data |
| `utils.py` | Helper functions |
| `configs/config.json` | All settings in one place |

### Folder Structure
```
Wildfire_Impact/
├── models/          ← 10 neural network architectures
├── configs/         ← Configuration files
├── data/processed/  ← Prepared datasets
├── results/         ← Training outputs & checkpoints
└── losses/          ← Loss functions (Dice, BCE)
```

---

## 4. Available Models (10 Total)

| Model | Type | Best For |
|-------|------|----------|
| **BAM-CD** | CNN | Overall best (project's flagship) |
| **U-Net** | CNN | Simple, reliable baseline |
| **ChangeFormer** | Transformer | Capturing global context |
| **SNUNet** | CNN | Dense feature connections |
| **FC-EF-Diff** | CNN | Fast, lightweight |

*Switch models by changing one line in config:* `"method": "bam_cd"`

---

## 5. Quick Start Commands

```bash
# 1. Generate test data (no downloads needed)
python create_synthetic_dataset.py --event_type wildfire

# 2. Train a model
python run_experiment.py --config configs/config.json

# 3. Evaluate
python run_experiment.py --config configs/config_eval_synthetic.json

# 4. Visualize results
python visualize_predictions.py --config configs/config.json --mode test
```

---

## 6. Key Configuration Options

```json
{
    "method": "bam_cd",           // Which model
    "event_type": "wildfire",     // or "drought"
    "mode": "train",              // or "eval"
    "train": {
        "n_epochs": 150,          // Training duration
        "batch_size": 8           // GPU memory usage
    }
}
```

---

## 7. Understanding the Output

### Metrics Explained
| Metric | Meaning |
|--------|---------|
| **F1-Score** | Balance of precision & recall (higher = better) |
| **IoU** | Overlap between prediction & truth (higher = better) |
| **Precision** | "Of what I predicted as burnt, how much actually was?" |
| **Recall** | "Of what actually burned, how much did I find?" |

### Output Files
```
results/<model>/<timestamp>/
├── checkpoints/           ← Saved model weights
│   └── best_segmentation.pt
└── logs/                  ← Training history
```

---

## 8. Technology Stack

| Tech | Why |
|------|-----|
| **Python** | ML ecosystem, readable |
| **PyTorch** | Flexible, easy debugging |
| **Sentinel-2** | Free satellite data, 10m resolution |
| **NumPy** | Fast array operations |
| **HDF5** | Efficient large data storage |

---

## 9. Common Bugs & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `KeyError: 'positive_flag'` | Old dataset format | Use `.get('positive_flag', False)` |
| Loss becomes `NaN` | Bad pixel values | Clamp/normalize input data |
| Poor eval performance | Forgot `model.eval()` | Always call before testing |
| GPU memory leak | Storing tensors | Use `.item()` for logging |
| No checkpoint saved | Val F1 never improved | Check data loading |

---

## 10. Best Practices Checklist

- [ ] Set random seeds for reproducibility
- [ ] Call `model.eval()` before testing
- [ ] Use `.item()` when logging losses
- [ ] Visualize data before training
- [ ] Start with synthetic data to test pipeline
- [ ] Use config files, not hardcoded values
- [ ] Save configs with results for reproducibility

---

## 11. Multi-Hazard Support

The project supports **two event types**:

| Event | Class 0 | Class 1 |
|-------|---------|---------|
| **Wildfire** | Unburnt | Burnt |
| **Drought** | No drought | Drought-affected |

Switch via config: `"event_type": "drought"`

---

## 12. Performance Tips

1. **Mixed Precision**: Set `"mixed_precision": true` → 2× faster
2. **Batch Size**: Increase until GPU is ~80% full
3. **Early Stopping**: Monitor val F1, stop if no improvement
4. **Data Augmentation**: Set `"augmentation": true` for more robust models

---

## 13. What to Read First (Learning Path)

```
Beginner (30 min):
README.md → QUICK_GUIDE.md (this file) → Run synthetic test

Intermediate (2 hours):
↑ above ↑ → FEATURES.md → EXPLANATION.md → Modify config & retrain

Advanced (4+ hours):
↑ above ↑ → Study models/bam_cd/ → Add new model → Read paper
```

---

## 14. Key Insights for Engineers

1. **Config-driven design** = Easy experimentation
2. **Siamese networks** = Fair image comparison
3. **Dice + CE loss** = Handles class imbalance
4. **Modular code** = Swap components easily
5. **Synthetic data** = Fast iteration during development

---

## 15. Resources

- **FLOGA Paper**: IEEE JSTARS 2024, DOI: 10.1109/JSTARS.2024.3381737
- **Dataset**: [Dropbox Link](https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h)
- **Full Documentation**: See `main.md` in this repo

---

**TL;DR**: Satellite images in → AI magic → Burnt area map out. Change one config line to try different models. Start with synthetic data to test everything works.

*Happy engineering! 🚀*
