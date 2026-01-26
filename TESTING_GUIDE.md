# 🔥 Wildfire Impact - Testing Commands

Quick copy-paste commands. Choose **Mac/Linux** or **Windows** section.

---

## 🍎 Mac / Linux Commands

### Setup (One-time)
```bash
cd Wildfire_Impact
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Quick Test (5 minutes)
```bash
# Generate synthetic data
python3 create_synthetic_dataset.py --event_type wildfire

# Run evaluation
python3 run_experiment.py --config configs/config_eval_synthetic.json
```

### Full Test Suite
```bash
# 1. Wildfire
python3 create_synthetic_dataset.py --event_type wildfire
python3 run_experiment.py --config configs/config_eval_synthetic.json

# 2. Drought
python3 create_synthetic_dataset.py --event_type drought
python3 run_experiment.py --config configs/config_eval_synthetic_drought.json

# 3. Visualize
python3 visualize_predictions.py --results_path results/viz/ --config configs/config_eval_synthetic.json --mode test
```

### Train Custom Model
```bash
python3 run_experiment.py --config configs/config.json
```

---

## 🪟 Windows Commands

### Setup (One-time)
```cmd
cd Wildfire_Impact
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Quick Test (5 minutes)
```cmd
:: Generate synthetic data
python create_synthetic_dataset.py --event_type wildfire

:: Run evaluation
python run_experiment.py --config configs/config_eval_synthetic.json
```

### Full Test Suite
```cmd
:: 1. Wildfire
python create_synthetic_dataset.py --event_type wildfire
python run_experiment.py --config configs/config_eval_synthetic.json

:: 2. Drought
python create_synthetic_dataset.py --event_type drought
python run_experiment.py --config configs/config_eval_synthetic_drought.json

:: 3. Visualize
python visualize_predictions.py --results_path results/viz/ --config configs/config_eval_synthetic.json --mode test
```

### Train Custom Model
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

**Note:** Class labels automatically switch based on event type:
- **Wildfire**: Burnt / Unburnt
- **Drought**: Drought-affected / No drought

---

x## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` (Mac) or `pip install -r requirements.txt` (Windows) |
| `CUDA out of memory` | Edit config: `"batch_size": 2` |
| `FileNotFoundError` | Run synthetic data creation first |
| `No module named 'timm'` | `pip3 install timm` (Mac) or `pip install timm` (Windows) |

---

## 📁 Config Files Reference

| Config | Purpose |
|--------|---------|
| `config_eval_synthetic.json` | Evaluate on synthetic wildfire data |
| `config_eval_synthetic_drought.json` | Evaluate on synthetic drought data |
| `config.json` | Train new model |
| `config_eval.json` | Evaluate on real data |

---

*Last Updated: January 27, 2026*
