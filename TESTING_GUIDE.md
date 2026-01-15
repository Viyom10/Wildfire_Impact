# FLOGA Testing Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Quick Start Testing](#quick-start-testing)
4. [Detailed Testing Procedures](#detailed-testing-procedures)
5. [Expected Outputs](#expected-outputs)
6. [Troubleshooting](#troubleshooting)
7. [Validation Checklist](#validation-checklist)

---

## 🔧 Prerequisites

### Required Software
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU training)
- **Git**: For version control

### Hardware Requirements
- **Minimum**: 16GB RAM, CPU-only training
- **Recommended**: 32GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 50GB+ free space for dataset and results

---

## 🚀 Environment Setup

### Step 1: Install Python Dependencies

```powershell
# Navigate to FLOGA directory
cd C:\Users\hp\OneDrive\Desktop\WildFire_Monitoring\FLOGA

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn tqdm
pip install h5py hdf5plugin
pip install pyjson5
pip install einops
pip install torchmetrics
pip install wandb  # Optional: for experiment tracking
pip install segmentation-models-pytorch  # For BAM-CD model
```

### Step 2: Verify Installation

```powershell
# Test Python imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import h5py, hdf5plugin; print('HDF5 support: OK')"
python -c "import pyjson5; print('pyjson5: OK')"
```

**Expected Output**:
```
PyTorch version: 2.x.x
CUDA available: True  # or False if CPU-only
HDF5 support: OK
pyjson5: OK
```

---

## ⚡ Quick Start Testing

### Option A: Test with Pre-trained Model (Fastest)

If you have access to pre-trained weights:

```powershell
# 1. Download pre-trained BAM-CD model weights
# Place in: models/pretrained/bam_cd_best.pt

# 2. Edit configs/config.json
# Set:
#   "mode": "eval"
#   "paths.load_state": "models/pretrained/bam_cd_best.pt"

# 3. Run evaluation
python run_experiment.py --config configs/config.json
```

### Option B: Test Dataset Creation (No Download Required)

Test the data preprocessing pipeline:

```powershell
# Create a small test with dummy data (if you have HDF5 files)
python create_dataset.py --help

# View available options
```

### Option C: Quick Training Test (Minimal)

Train for a few epochs to verify the pipeline:

```powershell
# 1. Edit configs/config.json
#    Set: "train.n_epochs": 2
#         "train.rep_times": 1
#         "datasets.batch_size": 2

# 2. Run training
python run_experiment.py --config configs/config.json
```

---

## 🧪 Detailed Testing Procedures

### Test 1: Dataset Preprocessing

**Purpose**: Verify data pipeline works correctly

#### Step 1.1: Download FLOGA Dataset

```powershell
# Download from: https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h
# Save .h5 files to: data/raw/
```

#### Step 1.2: Create Analysis-Ready Dataset

```powershell
# Stage 1: Export patches
python create_dataset.py `
    --floga_path data/raw/ `
    --out_path data/processed/ `
    --out_size 256 256 `
    --stages 1 `
    --sea_ratio 0.9 `
    --out_format numpy
```

**Expected Output**:
```
--- STAGE 1 ---
(1/5) FLOGA_2017_sen2_20_mod_500.h5: 100%
...
Exported patches to data/processed/2017/
```

#### Step 1.3: Create Train/Val/Test Splits

```powershell
python create_dataset.py `
    --floga_path data/raw/ `
    --out_path data/processed/ `
    --out_size 256 256 `
    --stages 2 `
    --ratio 60 20 20 `
    --sample 1 `
    --random_seed 999
```

**Expected Output**:
```
--- STAGE 2 ---
Found 326 events to export.
Splitting train/val/test sets...
Splitting into 195 train, 66 val and 65 test events...
```

#### Step 1.4: Verify Created Files

```powershell
# Check processed data
ls data/processed/

# Should see:
# - Folders: 2017/, 2018/, 2019/, 2020/, 2021/
# - Files: sen2_20_mod_500/train.pkl, val.pkl, test.pkl
```

---

### Test 2: Configuration Validation

**Purpose**: Ensure configuration files are valid

```powershell
# Test configuration loading
python -c "import pyjson5; cfg = pyjson5.load(open('configs/config.json')); print('Config loaded successfully')"

# Test all method configs
Get-ChildItem configs/method/*.json | ForEach-Object {
    python -c "import pyjson5; pyjson5.load(open('$_')); print('✓ $($_.Name)')"
}
```

**Expected Output**:
```
Config loaded successfully
✓ bam_cd.json
✓ unet.json
✓ snunet.json
...
```

---

### Test 3: Dataset Loading

**Purpose**: Verify dataset class works correctly

Create test script `test_dataset.py`:

```python
import sys
sys.path.append('.')
import pyjson5
from pathlib import Path
from dataset_utils import Dataset
from torch.utils.data import DataLoader

# Load config
configs = pyjson5.load(open('configs/config.json'))

# Update paths
configs['paths']['dataset'] = 'data/processed/'
configs['dataset_type'] = 'sen2_20_mod_500'

print("Testing Dataset Loading...")

# Test train dataset
train_dataset = Dataset('train', configs)
print(f"✓ Train dataset: {len(train_dataset)} samples")

# Test validation dataset
val_dataset = Dataset('val', configs)
print(f"✓ Val dataset: {len(val_dataset)} samples")

# Test data loader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
batch = next(iter(train_loader))

print(f"✓ Batch keys: {batch.keys()}")
print(f"✓ Before image shape: {batch['S2_before_image'].shape}")
print(f"✓ After image shape: {batch['S2_after_image'].shape}")
print(f"✓ Label shape: {batch['label'].shape}")

print("\n✅ Dataset loading test PASSED!")
```

Run test:
```powershell
python test_dataset.py
```

**Expected Output**:
```
Testing Dataset Loading...
✓ Train dataset: 15000 samples
✓ Val dataset: 5000 samples
✓ Batch keys: dict_keys(['S2_before_image', 'S2_after_image', 'label', ...])
✓ Before image shape: torch.Size([2, 9, 256, 256])
✓ After image shape: torch.Size([2, 9, 256, 256])
✓ Label shape: torch.Size([2, 256, 256])

✅ Dataset loading test PASSED!
```

---

### Test 4: Model Initialization

**Purpose**: Verify all models can be initialized

Create test script `test_models.py`:

```python
import sys
sys.path.append('.')
import torch
import pyjson5
from pathlib import Path
from utils import init_model

models_to_test = [
    'bam_cd', 'unet', 'fc_ef_conc', 'fc_ef_diff', 
    'snunet', 'hfanet', 'changeformer', 'bit_cd',
    'adhr_cdnet', 'transunet_cd'
]

# Base config
base_config = pyjson5.load(open('configs/config.json'))

print("Testing Model Initialization...\n")

for model_name in models_to_test:
    try:
        # Load model config
        model_config = pyjson5.load(open(f'configs/method/{model_name}.json'))
        
        # Update config
        config = base_config.copy()
        config['method'] = model_name
        
        # Initialize model
        model = init_model(config, model_config, None, 9, 'cpu')
        
        # Test forward pass
        x1 = torch.randn(1, 9, 256, 256)
        x2 = torch.randn(1, 9, 256, 256)
        
        with torch.no_grad():
            output = model[0](x1, x2)
        
        print(f"✓ {model_name:15s} - Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ {model_name:15s} - ERROR: {str(e)}")

print("\n✅ Model initialization test completed!")
```

Run test:
```powershell
python test_models.py
```

**Expected Output**:
```
Testing Model Initialization...

✓ bam_cd          - Output shape: torch.Size([1, 2, 256, 256])
✓ unet            - Output shape: torch.Size([1, 2, 256, 256])
✓ fc_ef_conc      - Output shape: torch.Size([1, 2, 256, 256])
...

✅ Model initialization test completed!
```

---

### Test 5: Training Pipeline (Short Run)

**Purpose**: Verify complete training pipeline

#### Step 5.1: Create Test Configuration

```powershell
# Copy base config for testing
Copy-Item configs/config.json configs/config_test.json
```

Edit `configs/config_test.json`:
```json
{
    "method": "unet",
    "mode": "train",
    "dataset_type": "sen2_20_mod_500",
    "gpu_ids": [0],  // or empty [] for CPU
    "train": {
        "n_epochs": 3,
        "rep_times": 1,
        "val_freq": 1,
        "save_checkpoint_freq": -1,
        "print_freq": 5,
        "mixed_precision": false,
        "loss_function": "cross_entropy",
        "weighted_loss": false,
        "resume": false,
        "log_landcover_metrics": false
    },
    "wandb": {
        "activate": false
    },
    "paths": {
        "dataset": "data/processed/",
        "results": "results_test/",
        "load_state": null
    },
    "datasets": {
        "train": "allEvents_60-20-20_r1_v4_train.pkl",
        "val": "allEvents_60-20-20_r1_v4_val.pkl",
        "test": "allEvents_60-20-20_r1_v4_test.pkl",
        "data_source": "sen2",
        "batch_size": 4,
        "num_workers": 2,
        "augmentation": false,
        "oversampling": false
    }
}
```

#### Step 5.2: Run Training Test

```powershell
python run_experiment.py --config configs/config_test.json
```

**Expected Output**:
```
--- Training a new model ---
--- Model path: results_test/unet/20260115123456 ---
Using cross_entropy with class weights: (1, 1).

===== REP 0 =====

(0) Train Loss: 0.6931: 100%|████████| 1875/1875
F1-score: 0.5234
VAL F1-score: 0.5123

(1) Train Loss: 0.6245: 100%|████████| 1875/1875
F1-score: 0.6234
VAL F1-score: 0.6123
Saved best checkpoint!

(2) Train Loss: 0.5891: 100%|████████| 1875/1875
F1-score: 0.6734
VAL F1-score: 0.6523
Saved best checkpoint!
```

#### Step 5.3: Verify Training Outputs

```powershell
# Check results directory
ls results_test/unet/

# Should see:
# - <timestamp>/
#   - checkpoints/
#     - 0/
#       - best_segmentation.pt
#       - best_segmentation.txt
```

---

### Test 6: Evaluation Pipeline

**Purpose**: Test model evaluation on test set

#### Step 6.1: Update Config for Evaluation

Edit `configs/config_test.json`:
```json
{
    "mode": "eval",
    "paths": {
        "load_state": "results_test/unet/<timestamp>/checkpoints/0/best_segmentation.pt"
    }
}
```

#### Step 6.2: Run Evaluation

```powershell
python run_experiment.py --config configs/config_test.json
```

**Expected Output**:
```
--- Testing model for results_test/unet/.../best_segmentation.pt ---

Loading checkpoint...

Evaluating: 100%|████████| 1250/1250

(0) test F-Score (Unburnt): 85.23
(0) test F-Score (Burnt): 78.45
(0) test IoU (Unburnt): 74.32
(0) test IoU (Burnt): 64.56
(0) test Precision (Unburnt): 86.12
(0) test Precision (Burnt): 76.23
(0) test Recall (Unburnt): 84.34
(0) test Recall (Burnt): 80.89
(0) test Accuracy (Unburnt): 88.56
(0) test Accuracy (Burnt): 82.34
(0) test MeanIoU: 69.44

===============

f1 (burnt): 78.45
f1 (unburnt): 85.23
Mean f-score: 81.84
Mean IoU: 69.44
```

---

### Test 7: Visualization

**Purpose**: Generate and verify prediction visualizations

```powershell
python visualize_predictions.py `
    --results_path results_test/visualizations/ `
    --config configs/config_test.json `
    --bands nrg `
    --mode test `
    --export_numpy_preds
```

**Expected Output**:
```
--- Loading model checkpoint: results_test/unet/.../best_segmentation.pt ---

Processing: 100%|████████| 1250/1250

Saved visualizations to:
- results_test/visualizations/test/TP/
- results_test/visualizations/test/FP/
- results_test/visualizations/test/FN/
```

Verify outputs:
```powershell
ls results_test/visualizations/test/

# Should see folders: TP/, FP/, FN/
# Each containing PNG images and optionally NPY files
```

---

## 📊 Expected Outputs

### After Dataset Creation:
```
data/processed/
├── 2017/
│   ├── *.label.npy
│   ├── *.S2_before.npy
│   ├── *.S2_after.npy
│   └── ... (cloud masks, sea masks, etc.)
├── 2018/
│   └── ...
├── sen2_20_mod_500/
│   ├── allEvents_60-20-20_r1_v4_train.pkl
│   ├── allEvents_60-20-20_r1_v4_val.pkl
│   └── allEvents_60-20-20_r1_v4_test.pkl
```

### After Training:
```
results/
└── unet/
    └── 20260115123456/
        ├── checkpoints/
        │   └── 0/
        │       ├── best_segmentation.pt
        │       └── best_segmentation.txt
        ├── wandb_id.json (if WandB enabled)
        └── 0 lrs.txt
```

### After Evaluation:
```
results/
└── unet/
    └── 20260115123456/
        └── lc_stats.csv (if log_landcover_metrics: true)
```

### After Visualization:
```
results_test/visualizations/
└── test/
    ├── TP/
    │   ├── event001_patch001.png
    │   └── event001_patch001.npy
    ├── FP/
    └── FN/
```

---

## 🔍 Troubleshooting

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```powershell
pip install <missing-module>
# or
pip install -r requirements.txt  # if available
```

### Issue 2: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```json
// Option 1: Reduce batch size
"datasets": {
    "batch_size": 2  // Reduce from 8
}

// Option 2: Use CPU
"gpu_ids": []

// Option 3: Enable mixed precision
"train": {
    "mixed_precision": true
}
```

### Issue 3: Dataset Path Not Found

**Error**: `The dataset path does not exist!`

**Solution**:
```json
// Update paths in configs/config.json
"paths": {
    "dataset": "C:/Users/hp/OneDrive/Desktop/WildFire_Monitoring/FLOGA/data/processed/"
}
```

### Issue 4: HDF5 Plugin Error

**Error**: `OSError: Can't read data (inflate() failed)`

**Solution**:
```powershell
pip install hdf5plugin
```

### Issue 5: NaN Loss During Training

**Possible Causes**:
- Learning rate too high
- Input data not normalized
- Model initialization issue

**Solutions**:
```json
// Reduce learning rate
{
    "optimizer": {
        "learning_rate": 0.0001  // Reduce from 0.001
    }
}

// Ensure data scaling
{
    "datasets": {
        "scale_input": "normalize"  // or "clamp_scale_10000"
    }
}
```

### Issue 6: Slow Data Loading

**Error**: Training very slow, CPU bottleneck

**Solution**:
```json
// Increase number of workers
"datasets": {
    "num_workers": 4  // Adjust based on CPU cores
}
```

---

## ✅ Validation Checklist

Use this checklist to verify your setup:

### Pre-Training Checklist:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (PyTorch, etc.)
- [ ] CUDA working (if using GPU)
- [ ] Dataset downloaded
- [ ] Dataset preprocessed (patches created)
- [ ] Train/val/test splits created
- [ ] Configuration file updated with correct paths
- [ ] Dataset loading test passed
- [ ] Model initialization test passed

### During Training Checklist:
- [ ] Training loss decreasing
- [ ] Validation F1-score improving
- [ ] No NaN values in loss
- [ ] GPU utilization reasonable (if using GPU)
- [ ] Checkpoints being saved

### Post-Training Checklist:
- [ ] Best checkpoint saved
- [ ] Evaluation metrics reasonable (F1 > 50%)
- [ ] Visualizations generated successfully
- [ ] Results reproducible

---

## 🎯 Quick Test Script

Save as `quick_test.py`:

```python
"""Quick test to verify FLOGA setup"""
import sys
import torch
from pathlib import Path

print("=" * 60)
print("FLOGA Quick Test Script")
print("=" * 60)

tests_passed = 0
tests_failed = 0

# Test 1: Python version
print("\n1. Testing Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    tests_passed += 1
else:
    print(f"   ✗ Python version too old: {sys.version_info}")
    tests_failed += 1

# Test 2: PyTorch
print("\n2. Testing PyTorch...")
try:
    print(f"   ✓ PyTorch {torch.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"   ✗ PyTorch import failed: {e}")
    tests_failed += 1

# Test 3: CUDA
print("\n3. Testing CUDA...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    tests_passed += 1
else:
    print(f"   ⚠ CUDA not available (CPU-only mode)")
    tests_passed += 1

# Test 4: Required imports
print("\n4. Testing required imports...")
required = ['numpy', 'pandas', 'h5py', 'hdf5plugin', 'pyjson5', 'einops', 'torchmetrics']
for module in required:
    try:
        __import__(module)
        print(f"   ✓ {module}")
        tests_passed += 1
    except ImportError:
        print(f"   ✗ {module} - NOT INSTALLED")
        tests_failed += 1

# Test 5: Config files
print("\n5. Testing config files...")
if Path('configs/config.json').exists():
    print(f"   ✓ configs/config.json")
    tests_passed += 1
else:
    print(f"   ✗ configs/config.json NOT FOUND")
    tests_failed += 1

# Test 6: Model files
print("\n6. Testing model files...")
model_files = ['unet.py', 'snunet.py', 'changeformer.py']
for mf in model_files:
    if Path(f'models/{mf}').exists():
        print(f"   ✓ models/{mf}")
        tests_passed += 1
    else:
        print(f"   ✗ models/{mf} NOT FOUND")
        tests_failed += 1

# Summary
print("\n" + "=" * 60)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
if tests_failed == 0:
    print("\n✅ ALL TESTS PASSED! Ready to run experiments.")
else:
    print(f"\n⚠ {tests_failed} test(s) failed. Please fix issues above.")
print("=" * 60)
```

Run it:
```powershell
python quick_test.py
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs**: Look at terminal output for error messages
2. **Verify paths**: Ensure all paths in config files are correct
3. **Check GPU memory**: Use `nvidia-smi` to monitor GPU usage
4. **Reduce complexity**: Start with smaller batch sizes and fewer epochs
5. **Test components individually**: Use the individual test scripts above

---

## 🎓 Next Steps

After successful testing:

1. **Full training**: Increase epochs to 150, rep_times to 10
2. **Experiment tracking**: Enable WandB for better monitoring
3. **Hyperparameter tuning**: Try different learning rates, batch sizes
4. **Model comparison**: Test all 10 models and compare results
5. **Publication-ready results**: Use the benchmark splits provided

---

*Last Updated: January 15, 2026*
