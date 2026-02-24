# 🔥 Wildfire Impact - Testing Commands

Quick copy-paste commands. Choose **🍎 Mac (python3)** or **🪟 Windows / Linux (python)** section.

---

## 🍎 Mac (python3) Commands

### Setup (One-time)
```bash
cd Wildfire_Impact
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### 1. Test on Dataset Directly (Synthetic Data)

Generate and evaluate on synthetic datasets for rapid testing.

```bash
# ---------- WILDFIRE HAZARD ----------
# Generate synthetic wildfire data
python3 create_synthetic_dataset.py --event_type wildfire
# Run evaluation
python3 run_experiment.py --config configs/config_eval_synthetic.json

# ---------- DROUGHT HAZARD ----------
# Generate synthetic drought data
python3 create_synthetic_dataset.py --event_type drought
# Run evaluation
python3 run_experiment.py --config configs/config_eval_synthetic_drought.json

# ---------- VISUALIZE DATASET RESULTS ----------
# Create images from the evaluation results
python3 visualize_predictions.py --results_path results/viz/ --config configs/config_eval_synthetic.json --mode test
```

### 2. Predict from Own Images (`predict_from_images.py`)

Run inference on any image pair (before / after). 

```bash
# ---------- WILDFIRE HAZARD (Default ML Model) ----------
python3 predict_from_images.py --before goa_test/goa_before.jpg --after goa_test/goa_after.jpg --rgb --hazard wildfire --output results/wildfire_result.png

# ---------- DROUGHT HAZARD (ML Model) ----------
python3 predict_from_images.py --before goa_test/goa_before.jpg --after goa_test/goa_after.jpg --rgb --hazard drought --output results/drought_result.png

# ---------- RAINFALL / FLOOD HAZARD (Spectral Index) ----------
python3 predict_from_images.py --before goa_test/goa_before.jpg --after goa_test/goa_after.jpg --rgb --hazard rainfall --output results/rainfall_result.png

# ---------- VEGETATION HEALTH HAZARD (Spectral Index) ----------
python3 predict_from_images.py --before goa_test/goa_before.jpg --after goa_test/goa_after.jpg --rgb --hazard vegetation --output results/vegetation_result.png

# Run a quick demo with sample data
python3 predict_from_images.py --demo
```

### 3. Train Custom Model
```bash
python3 run_experiment.py --config configs/config.json
```

---

## 🪟 Windows / Linux (python) Commands

### Setup (One-time)
```cmd
cd Wildfire_Impact
python -m venv venv
:: For Windows: venv\Scripts\activate
:: For Linux:  source venv/bin/activate
pip install -r requirements.txt
```

### 1. Test on Dataset Directly (Synthetic Data)

Generate and evaluate on synthetic datasets for rapid testing.

```cmd
:: ---------- WILDFIRE HAZARD ----------
:: Generate synthetic wildfire data
python create_synthetic_dataset.py --event_type wildfire
:: Run evaluation
python run_experiment.py --config configs/config_eval_synthetic.json

:: ---------- DROUGHT HAZARD ----------
:: Generate synthetic drought data
python create_synthetic_dataset.py --event_type drought
:: Run evaluation
python run_experiment.py --config configs/config_eval_synthetic_drought.json

:: ---------- VISUALIZE DATASET RESULTS ----------
:: Create images from the evaluation results
python visualize_predictions.py --results_path results/viz/ --config configs/config_eval_synthetic.json --mode test
```

### 2. Predict from Own Images (`predict_from_images.py`)

Run inference on any image pair (before / after). 

```cmd
:: ---------- WILDFIRE HAZARD (Default ML Model) ----------
python predict_from_images.py --before goa_test\goa_before.jpg --after goa_test\goa_after.jpg --rgb --hazard wildfire --output results\wildfire_result.png

:: ---------- DROUGHT HAZARD (ML Model) ----------
python predict_from_images.py --before goa_test\goa_before.jpg --after goa_test\goa_after.jpg --rgb --hazard drought --output results\drought_result.png

:: ---------- RAINFALL / FLOOD HAZARD (Spectral Index) ----------
python predict_from_images.py --before goa_test\goa_before.jpg --after goa_test\goa_after.jpg --rgb --hazard rainfall --output results\rainfall_result.png

:: ---------- VEGETATION HEALTH HAZARD (Spectral Index) ----------
python predict_from_images.py --before goa_test\goa_before.jpg --after goa_test\goa_after.jpg --rgb --hazard vegetation --output results\vegetation_result.png

:: Run a quick demo with sample data
python predict_from_images.py --demo
```

### 3. Train Custom Model
```cmd
python run_experiment.py --config configs/config.json
```

---

## ✅ Expected Output

### Wildfire - Synthetic Data Creation:
```
============================================================
✅ Synthetic wildfire dataset created successfully!
============================================================
Total patches: 16
```

### Wildfire - Evaluation:
```
Using event type: wildfire
test Loss: 3.2042: 100%|█████████████████████████████████| 2/2
(0) test F-Score (Unburnt): 93.01
(0) test F-Score (Burnt): xx.xx
(0) test IoU (Unburnt): 86.93
Mean IoU: 43.46
```

### Drought - Synthetic Data Creation:
```
============================================================
✅ Synthetic drought dataset created successfully!
============================================================
Total patches: 16
```

### Drought - Evaluation:
```
Using event type: drought
test Loss: 4.9020: 100%|█████████████████████████████████| 2/2
(0) test F-Score (No drought): 90.91
(0) test F-Score (Drought-affected): xx.xx
(0) test IoU (No drought): 83.33
Mean IoU: 41.67
```

**Note:** Class labels automatically switch based on event type.

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` (Mac) or `pip install -r requirements.txt` (Windows/Linux) |
| `CUDA out of memory` | Edit config: `"batch_size": 2` |
| `FileNotFoundError` | Run synthetic data creation first |
| `No module named 'timm'` | `pip3 install timm` (Mac) or `pip install timm` (Windows/Linux) |
| `unrecognized arguments: --hazard` | Ensure you have the latest script version |

---

## 📁 Config Files Reference

| Config | Purpose |
|--------|---------|
| `config_eval_synthetic.json` | Evaluate on synthetic wildfire data |
| `config_eval_synthetic_drought.json` | Evaluate on synthetic drought data |
| `config.json` | Train new model |
| `config_eval.json` | Evaluate on real data |

---

## 🖼️ Predict From Images Options Reference

| Option | Description |
|--------|-------------|
| `--before` | Path to before image (required) |
| `--after` | Path to after image (required) |
| `--hazard` | Choose from: `wildfire`, `drought`, `rainfall`, `vegetation` |
| `--rgb` | Use when input is RGB (3-channel) images |
| `--output` | Path to save visualization |
| `--output-mask` | Path to save binary prediction mask (.npy) |
| `--output-prob` | Path to save probability map (.npy) |
| `--threshold` | Probability threshold (default: 0.5) |
| `--no-show` | Don't display visualization window |
| `--demo` | Run demo with synthetic data |

### Supported Image Formats
- **GeoTIFF** (.tif, .tiff) - Preferred for satellite imagery
- **NumPy arrays** (.npy) - Direct array input
- **Standard images** (.png, .jpg, .jpeg) - Use with `--rgb` flag

---

*Last Updated: 2026-02-25*
