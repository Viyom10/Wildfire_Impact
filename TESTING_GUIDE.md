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

### Option A: Test with Synthetic Data (Fastest - No Downloads)

Test the complete pipeline with minimal synthetic data:

```powershell
# 1. Generate synthetic wildfire dataset
python create_synthetic_dataset.py --event_type wildfire

# 2. Verify dataset was created
ls data/processed/sen2_20_mod_500/
# Should show: synthetic_train.pkl, synthetic_val.pkl, synthetic_test.pkl

# 3. Run quick training test
python run_experiment.py --config configs/config_eval_synthetic.json
```

### Option B: Test with Pre-trained Model (Fastest)

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

### Test 0: Synthetic Data Generation (NEW - Multi-Hazard)

**Purpose**: Test synthetic dataset creation for wildfire and drought detection

#### Step 0.1: Generate Wildfire Synthetic Data

```powershell
python create_synthetic_dataset.py --event_type wildfire
```

**Expected Output**:
```
============================================================
Creating Minimal Test Dataset
============================================================

Creating synthetic patches...

Creating train split (10 samples)...
  ✓ Created synthetic_event_train_000_patch_000 (positive)
  ✓ Created synthetic_event_train_001_patch_000 (negative)
  ...
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_train.pkl

Creating val split (3 samples)...
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_val.pkl

Creating test split (3 samples)...
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_test.pkl

============================================================
✅ Synthetic wildfire dataset created successfully!
============================================================

Dataset location: data/processed
Split files:
  - data/processed/sen2_20_mod_500/synthetic_train.pkl
  - data/processed/sen2_20_mod_500/synthetic_val.pkl
  - data/processed/sen2_20_mod_500/synthetic_test.pkl
```

#### Step 0.2: Generate Drought Synthetic Data

```powershell
python create_synthetic_dataset.py --event_type drought
```

**Expected Output**: (Similar to above, but drought patches have rectangular affected areas)
```
============================================================
✅ Synthetic drought dataset created successfully!
============================================================
```

#### Step 0.3: Verify Synthetic Data Files

```powershell
# Check that data was created
ls data/processed/2024/*.npy

# Should see multiple .npy files for before/after images and labels:
# - synthetic_event_train_000_patch_000.S2_before.npy
# - synthetic_event_train_000_patch_000.S2_after.npy
# - synthetic_event_train_000_patch_000.label.npy
# - synthetic_event_train_000_patch_000.clc_mask.npy
```

---

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

### Test 2: Configuration Validation (Updated)

**Purpose**: Ensure configuration files are valid and support event_type

```powershell
# Test configuration loading with event_type
python -c "import pyjson5; cfg = pyjson5.load(open('configs/config.json')); print(f'Config loaded successfully'); print(f'Event type: {cfg.get(\"event_type\", \"wildfire\")}')"

# Test all method configs
Get-ChildItem configs/method/*.json | ForEach-Object {
    python -c "import pyjson5; pyjson5.load(open('$_')); print('✓ $($_.Name)')"
}

# Verify event_type options are present in main config
python -c "
import pyjson5
cfg = pyjson5.load(open('configs/config.json'))
print(f'Event type in config: {cfg.get(\"event_type\", \"NOT FOUND\")}')
print(f'Supported types: wildfire, drought')
"
```

**Expected Output**:
```
Config loaded successfully
Event type: wildfire
✓ bam_cd.json
✓ unet.json
✓ snunet.json
...
Event type in config: wildfire
Supported types: wildfire, drought
```

---

### Test 3: Dataset Loading (Updated for Multi-Hazard)

**Purpose**: Verify dataset class works correctly with event_type filtering

Create test script `test_dataset_multihazard.py`:

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

print("Testing Multi-Hazard Dataset Loading...\n")

# Test 1: Wildfire dataset
print("Test 1: Loading wildfire dataset")
configs['event_type'] = 'wildfire'
try:
    wildfire_dataset = Dataset('train', configs)
    print(f"✓ Wildfire train dataset: {len(wildfire_dataset)} samples")
    
    # Check if dataset has event_type metadata
    batch = wildfire_dataset[0]
    if 'event_type' in batch:
        print(f"✓ Event type in batch: {batch['event_type']}")
except Exception as e:
    print(f"✗ Error loading wildfire dataset: {e}")

# Test 2: Drought dataset
print("\nTest 2: Loading drought dataset")
configs['event_type'] = 'drought'
try:
    drought_dataset = Dataset('train', configs)
    print(f"✓ Drought train dataset: {len(drought_dataset)} samples")
    
    # Check if dataset has event_type metadata
    batch = drought_dataset[0]
    if 'event_type' in batch:
        print(f"✓ Event type in batch: {batch['event_type']}")
except Exception as e:
    # Expected if drought data not available
    print(f"⚠ Drought dataset not found (expected if only wildfire data exists): {e}")

# Test 3: DataLoader with batch loading
print("\nTest 3: DataLoader functionality")
configs['event_type'] = 'wildfire'
train_dataset = Dataset('train', configs)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
batch = next(iter(train_loader))

print(f"✓ Batch keys: {batch.keys()}")
print(f"✓ Before image shape: {batch['S2_before_image'].shape}")
print(f"✓ After image shape: {batch['S2_after_image'].shape}")
print(f"✓ Label shape: {batch['label'].shape}")

print("\n✅ Multi-hazard dataset loading test PASSED!")
```

Run test:
```powershell
python test_dataset_multihazard.py
```

**Expected Output**:
```
Testing Multi-Hazard Dataset Loading...

Test 1: Loading wildfire dataset
✓ Wildfire train dataset: 16 samples
✓ Event type in batch: wildfire

Test 2: Loading drought dataset
⚠ Drought dataset not found (expected if only wildfire data exists)

Test 3: DataLoader functionality
✓ Batch keys: dict_keys(['S2_before_image', 'S2_after_image', 'label', ...])
✓ Before image shape: torch.Size([2, 9, 256, 256])
✓ After image shape: torch.Size([2, 9, 256, 256])
✓ Label shape: torch.Size([2, 256, 256])

✅ Multi-hazard dataset loading test PASSED!
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

### Test 5: Training Pipeline (Short Run) - Multi-Hazard

**Purpose**: Verify complete training pipeline works for wildfire and drought

#### Step 5.1: Create Test Configuration for Wildfire

```powershell
Copy-Item configs/config.json configs/config_test_wildfire.json
```

Edit `configs/config_test_wildfire.json`:
```json
{
    "method": "unet",
    "mode": "train",
    "event_type": "wildfire",
    "dataset_type": "sen2_20_mod_500",
    "gpu_ids": [0],
    "train": {
        "n_epochs": 2,
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
        "results": "results_test_wildfire/",
        "load_state": null
    },
    "datasets": {
        "train": "synthetic_train.pkl",
        "val": "synthetic_val.pkl",
        "test": "synthetic_test.pkl",
        "data_source": "sen2",
        "batch_size": 4,
        "num_workers": 2,
        "augmentation": false,
        "oversampling": false
    }
}
```

#### Step 5.2: Create Test Configuration for Drought

```powershell
Copy-Item configs/config_test_wildfire.json configs/config_test_drought.json
```

Edit `configs/config_test_drought.json` - Change:
```json
{
    "event_type": "drought",
    "paths": {
        "results": "results_test_drought/"
    }
}
```

#### Step 5.3: Run Wildfire Training Test

```powershell
python run_experiment.py --config configs/config_test_wildfire.json
```

**Expected Output**:
```
--- Training a new model ---
--- Model path: results_test_wildfire/unet/20260124123456 ---
Using event type: wildfire
Using cross_entropy with class weights: (1, 1).

===== REP 0 =====

(0) Train Loss: 0.6931: 100%|████████| 4/4
F1-score: 0.5234
VAL F1-score (Unburnt): 0.5123
VAL F1-score (Burnt): 0.4234

(1) Train Loss: 0.5891: 100%|████████| 4/4
F1-score: 0.6734
VAL F1-score (Unburnt): 0.6523
VAL F1-score (Burnt): 0.5892
Saved best checkpoint!
```

#### Step 5.4: Run Drought Training Test

```powershell
python run_experiment.py --config configs/config_test_drought.json
```

**Expected Output**:
```
--- Training a new model ---
--- Model path: results_test_drought/unet/20260124123457 ---
Using event type: drought
Using cross_entropy with class weights: (1, 1).

===== REP 0 =====
... (Similar to wildfire but with drought labels)
```

#### Step 5.5: Verify Training Outputs


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

### Test 6: Evaluation Pipeline (Automatic Label Detection)

**Purpose**: Test model evaluation on test set with automatic event_type detection

#### Step 6.1: Update Config for Wildfire Evaluation

Edit `configs/config_test_wildfire.json`:
```json
{
    "mode": "eval",
    "event_type": "wildfire",
    "paths": {
        "load_state": "results_test_wildfire/unet/<timestamp>/checkpoints/0/best_segmentation.pt"
    }
}
```

#### Step 6.2: Run Wildfire Evaluation

```powershell
python run_experiment.py --config configs/config_test_wildfire.json
```

**Expected Output**:
```
--- Testing model for results_test_wildfire/unet/.../best_segmentation.pt ---
Using event type: wildfire

Loading checkpoint...

Evaluating: 100%|████████| 2/2

(0) test F-Score (Unburnt): 85.23
(0) test F-Score (Burnt): 78.45
(0) test IoU (Unburnt): 74.32
(0) test IoU (Burnt): 64.56
...
```

#### Step 6.3: Run Drought Evaluation

Update `configs/config_test_drought.json` and run:

```powershell
python run_experiment.py --config configs/config_test_drought.json
```

**Expected Output**:
```
--- Testing model for results_test_drought/unet/.../best_segmentation.pt ---
Using event type: drought

Loading checkpoint...

Evaluating: 100%|████████| 2/2

(0) test F-Score (No drought): 85.23
(0) test F-Score (Drought-affected): 78.45
(0) test IoU (No drought): 74.32
(0) test IoU (Drought-affected): 64.56
...
```

**Note**: Class labels automatically change based on `event_type` in config!

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

### Multi-Hazard Testing Checklist (NEW):
- [ ] Synthetic wildfire dataset created successfully
- [ ] Synthetic drought dataset created successfully
- [ ] `event_type` parameter present in configs/config.json
- [ ] Wildfire training runs and produces loss outputs
- [ ] Drought training runs and produces loss outputs
- [ ] Wildfire metrics show correct labels (Unburnt/Burnt)
- [ ] Drought metrics show correct labels (No drought/Drought-affected)
- [ ] Class labels automatically change based on event_type

### Pre-Training Checklist:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (PyTorch, etc.)
- [ ] CUDA working (if using GPU)
- [ ] Dataset downloaded (or synthetic data generated)
- [ ] Dataset preprocessed (patches created)
- [ ] Train/val/test splits created
- [ ] Configuration file updated with correct paths
- [ ] `event_type` set to "wildfire" or "drought"
- [ ] Dataset loading test passed
- [ ] Model initialization test passed

### During Training Checklist:
- [ ] Event type logged at start ("Using event type: ...")
- [ ] Training loss decreasing
- [ ] Validation F1-score improving
- [ ] No NaN values in loss
- [ ] GPU utilization reasonable (if using GPU)
- [ ] Checkpoints being saved
- [ ] Correct class labels in wandb logs

### Post-Training Checklist:
- [ ] Best checkpoint saved
- [ ] Evaluation metrics reasonable (F1 > 50%)
- [ ] Class labels match event_type (Burnt/Unburnt or Drought/No drought)
- [ ] Visualizations generated successfully
- [ ] Results reproducible
- [ ] Both wildfire and drought models work correctly

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

# Test 5: Config event_type parameter (NEW)
print("\n5. Testing event_type in config...")
try:
    cfg = pyjson5.load(open('configs/config.json'))
    event_type = cfg.get('event_type', 'wildfire')
    if event_type in ['wildfire', 'drought']:
        print(f"   ✓ event_type: {event_type}")
        tests_passed += 1
    else:
        print(f"   ✗ Invalid event_type: {event_type}")
        tests_failed += 1
except Exception as e:
    print(f"   ✗ Error reading event_type: {e}")
    tests_failed += 1

# Test 6: Config files
print("\n6. Testing config files...")
if Path('configs/config.json').exists():
    print(f"   ✓ configs/config.json")
    tests_passed += 1
else:
    print(f"   ✗ configs/config.json NOT FOUND")
    tests_failed += 1

# Test 7: Model files
print("\n7. Testing model files...")
model_files = ['unet.py', 'snunet.py', 'changeformer.py']
for mf in model_files:
    if Path(f'models/{mf}').exists():
        print(f"   ✓ models/{mf}")
        tests_passed += 1
    else:
        print(f"   ✗ models/{mf} NOT FOUND")
        tests_failed += 1

# Test 8: Dataset and experiment scripts (NEW)
print("\n8. Testing key scripts...")
scripts = ['create_synthetic_dataset.py', 'run_experiment.py', 'dataset_utils.py']
for script in scripts:
    if Path(script).exists():
        print(f"   ✓ {script}")
        tests_passed += 1
    else:
        print(f"   ✗ {script} NOT FOUND")
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

## � Complete End-to-End Test (Recommended Starting Point)

Follow these steps to test the entire system in 10-15 minutes:

### Step 1: Create Synthetic Wildfire Data (2 min)

```powershell
python create_synthetic_dataset.py --event_type wildfire
```

### Step 2: Create Synthetic Drought Data (2 min)

```powershell
python create_synthetic_dataset.py --event_type drought
```

### Step 3: Verify Synthetic Data (1 min)

```powershell
python test_dataset_multihazard.py
```

### Step 4: Run Quick Training Test - Wildfire (5 min)

```powershell
python run_experiment.py --config configs/config_eval_synthetic.json
```

Expected: Training completes with loss decreasing and F1-score improving

### Step 5: Verify Different Event Types Work (2 min)

Edit `configs/config_eval_synthetic.json` and change:
```json
"event_type": "drought"
```

Run again:
```powershell
python run_experiment.py --config configs/config_eval_synthetic.json
```

### Step 6: Check Class Labels Changed (1 min)

Monitor output and verify:
- **Wildfire**: "Using event type: wildfire", metrics show "Burnt/Unburnt"
- **Drought**: "Using event type: drought", metrics show "Drought-affected/No drought"

---

## 📝 Running Tests in Sequence

### Option 1: Minimal Test (5-10 minutes)

```powershell
# 1. Generate synthetic data
python create_synthetic_dataset.py --event_type wildfire

# 2. Run quick test with synthetic data
python quick_test.py

# 3. Train for 2 epochs on synthetic data
python run_experiment.py --config configs/config_eval_synthetic.json
```

### Option 2: Complete Multi-Hazard Test (15-20 minutes)

```powershell
# 1. Test synthetic wildfire
python create_synthetic_dataset.py --event_type wildfire
python run_experiment.py --config configs/config_test_wildfire.json

# 2. Test synthetic drought
python create_synthetic_dataset.py --event_type drought
python run_experiment.py --config configs/config_test_drought.json

# 3. Verify both worked
python test_dataset_multihazard.py
```

### Option 3: Comprehensive Validation (30+ minutes)

```powershell
# Run all validation steps
python quick_test.py
python test_models.py
python test_dataset_multihazard.py
python create_synthetic_dataset.py --event_type wildfire
python create_synthetic_dataset.py --event_type drought
python run_experiment.py --config configs/config_test_wildfire.json
python run_experiment.py --config configs/config_test_drought.json
python visualize_predictions.py --results_path results_test_wildfire/ --config configs/config_test_wildfire.json --mode test
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs**: Look at terminal output for error messages
2. **Run quick_test.py**: Verify basic setup is correct
3. **Verify paths**: Ensure all paths in config files are correct
4. **Check GPU memory**: Use `nvidia-smi` to monitor GPU usage
5. **Reduce complexity**: Start with smaller batch sizes and fewer epochs
6. **Test components individually**: Use the individual test scripts above
7. **Check event_type**: Verify `"event_type"` is set in your config file

### Common Issues:

| Issue | Solution |
|-------|----------|
| `KeyError: 'event_type'` | Add `"event_type": "wildfire"` to your config.json |
| Dataset not found | Verify `event_type` matches dataset metadata |
| Wrong class labels | Check that `event_type` in config matches your data type |
| No synthetic data | Run `python create_synthetic_dataset.py --event_type wildfire` first |
| CUDA out of memory | Reduce `batch_size` in config or set `gpu_ids: []` to use CPU |

---

## 🎓 Next Steps

After successful testing:

1. **Full training**: Increase epochs to 150, rep_times to 10
2. **Experiment tracking**: Enable WandB for better monitoring
3. **Both event types**: Train models on both wildfire and drought
4. **Hyperparameter tuning**: Try different learning rates, batch sizes
5. **Model comparison**: Test all 10 models and compare results
6. **Publication-ready results**: Use the benchmark splits provided

---

*Last Updated: January 24, 2026*
*Added multi-hazard (wildfire + drought) testing support*
