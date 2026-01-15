# FLOGA Project - Comprehensive Code Explanation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset: FLOGA](#dataset-floga)
3. [Architecture & Code Structure](#architecture--code-structure)
4. [Main Scripts](#main-scripts)
5. [Models & Architectures](#models--architectures)
6. [Data Pipeline](#data-pipeline)
7. [Training & Evaluation](#training--evaluation)
8. [Configuration System](#configuration-system)
9. [Loss Functions](#loss-functions)
10. [Usage Guide](#usage-guide)

---

## 🎯 Project Overview

**FLOGA** is a machine learning ready dataset and benchmark for **burnt area mapping** using satellite imagery. The project implements multiple state-of-the-art deep learning models for **change detection** to identify areas affected by wildfires.

### Key Features:
- **Task**: Binary change detection (burnt vs. unburnt areas)
- **Data Sources**: Sentinel-2 and MODIS satellite imagery
- **Input**: Bi-temporal images (before and after fire events)
- **Output**: Pixel-wise classification maps
- **Region**: Greece (326 wildfire events, 2017-2021)
- **Ground Truth**: High-resolution burnt area mappings from Hellenic Fire Service

---

## 🛰️ Dataset: FLOGA

### Data Structure
The FLOGA dataset contains aligned Sentinel-2 and MODIS imagery for wildfire events with the following components:

1. **Imagery**:
   - **Pre-fire** Sentinel-2/MODIS imagery
   - **Post-fire** Sentinel-2/MODIS imagery

2. **Masks**:
   - **Cloud masks** (pre-fire and post-fire)
   - **Water mask** (sea areas)
   - **Corine Land Cover mask** (land use classification)

3. **Labels**:
   - `0` = Non-burnt pixels
   - `1` = Burnt pixels
   - `2` = Pixels burnt in other fire events (same year) - **excluded from training/evaluation**

### Satellite Data Specifications

#### Sentinel-2 Bands Available:
- **10m resolution**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- **20m resolution**: B02-B07, B11, B12, B8A (9 bands total)
- **60m resolution**: B01-B07, B09, B11, B12, B8A (11 bands total)

#### MODIS Bands Available:
- **500m resolution**: B01-B07 (7 bands total)

### Data Format
- Original format: **HDF5 files** (compressed with BZip2)
- Processed format: **NumPy arrays (.npy)** or **PyTorch tensors (.pt)**
- Patch size: Configurable (default 256x256 pixels)

---

## 🏗️ Architecture & Code Structure

```
FLOGA/
├── configs/
│   ├── config.json              # Main configuration file
│   └── method/                  # Model-specific configurations
│       ├── bam_cd.json          # BAM-CD model config
│       ├── unet.json            # U-Net config
│       ├── snunet.json          # SNUNet config
│       ├── changeformer.json    # ChangeFormer config
│       └── ... (other models)
│
├── models/                      # Model architectures
│   ├── bam_cd/                  # BAM-CD (proposed model)
│   │   ├── model.py
│   │   ├── decoder.py
│   │   ├── encoders_base.py
│   │   └── ...
│   ├── unet.py                  # U-Net architecture
│   ├── snunet.py               # SNUNet architecture
│   ├── changeformer.py         # ChangeFormer architecture
│   └── ... (other models)
│
├── losses/                      # Loss functions
│   ├── bce_and_dice.py         # Combined BCE + Dice loss
│   └── dice.py                 # Dice loss
│
├── Main Scripts:
├── create_dataset.py           # Dataset preprocessing
├── run_experiment.py           # Training & evaluation
├── visualize_predictions.py    # Visualization tools
├── utils.py                    # Utility functions
├── dataset_utils.py            # Dataset loading & augmentation
└── cd_experiments_utils.py     # Training/evaluation loops
```

---

## 📜 Main Scripts

### 1. `create_dataset.py`
**Purpose**: Converts raw HDF5 files into analysis-ready dataset

**What it does**:
1. **Stage 1 - Export Patches**:
   - Reads HDF5 files containing satellite imagery
   - Crops images into smaller patches (e.g., 256×256)
   - Applies padding if necessary
   - Filters out sea patches (configurable sea:land ratio)
   - Exports patches as NumPy or PyTorch files

2. **Stage 2 - Data Splitting**:
   - Splits data into train/validation/test sets
   - Two modes: 
     - **Event-based**: Random split across all years
     - **Year-based**: Specific years for train/test
   - Handles class imbalance with negative sampling
   - Creates pickle files with metadata for each split

**Key Functions**:
- `export_patches()`: Main export function
- `get_padding_offset()`: Calculates padding for uneven divisions
- `sample_negatives()`: Balances positive/negative samples
- `export_csv_with_patch_paths()`: Creates split metadata

**Usage Example**:
```bash
python create_dataset.py \
    --floga_path path/to/hdf/files \
    --out_path data/ \
    --out_size 256 256 \
    --sample 1 \
    --ratio 60 20 20
```

---

### 2. `run_experiment.py`
**Purpose**: Main script for training and evaluating models

**Workflow**:

**Training Mode** (`mode: "train"`):
1. Load configuration files
2. Initialize dataset and dataloaders
3. Initialize model architecture
4. Setup optimizer and learning rate scheduler
5. Setup loss function with class weights
6. Train model with validation
7. Save best checkpoint based on validation F1-score
8. Log metrics to Weights & Biases (optional)

**Evaluation Mode** (`mode: "eval"`):
1. Load trained model checkpoint
2. Run inference on test set
3. Calculate comprehensive metrics
4. Generate land cover-specific statistics
5. Save results and visualizations

**Key Features**:
- **Multi-repetition experiments**: Run same experiment multiple times (`rep_times`)
- **Mixed precision training**: Faster training with AMP
- **Checkpoint management**: Auto-save best models
- **WandB integration**: Track experiments online
- **Land cover metrics**: Performance breakdown by land type

---

### 3. `visualize_predictions.py`
**Purpose**: Generate visual predictions and export results

**Features**:
- Load trained models
- Generate predictions on train/val/test sets
- Create visualizations with:
  - RGB or NRG (NIR-Red-Green) composites
  - Ground truth overlays
  - Prediction overlays
  - Error analysis (TP, FP, FN regions)
- Export predictions as NumPy arrays
- Separate results by prediction quality

**Usage Example**:
```bash
python visualize_predictions.py \
    --results_path results/visualization \
    --config configs/config.json \
    --bands nrg \
    --mode test
```

---

### 4. `utils.py`
**Purpose**: Core utility functions used throughout the project

**Key Functions**:

1. **Model Initialization** (`init_model`):
   - Factory function to create model instances
   - Supports 10 different architectures
   - Loads pre-trained weights if available
   - Handles device placement (CPU/GPU)

2. **Path Management**:
   - `init_model_log_path()`: Creates result directories
   - `resume_or_start()`: Handles checkpoint loading/resuming

3. **Loss Functions** (`create_loss`):
   - Cross-entropy with class weights
   - Focal loss (for imbalanced data)
   - Dice loss (for segmentation)
   - Combined BCE + Dice loss

4. **Optimizers** (`init_optimizer`):
   - Adam, AdamW, SGD
   - Configurable learning rate and weight decay

5. **Learning Rate Schedulers** (`init_lr_scheduler`):
   - Cosine annealing
   - Step decay
   - Linear decay
   - Constant learning rate

6. **Class Weights** (`compute_class_weights`):
   - Calculates weights based on class distribution
   - Addresses class imbalance in dataset

7. **Custom Metrics**:
   - `MyConfusionMatrix`: Custom confusion matrix implementation
   - `LandCoverMetrics`: Metrics per land cover type

8. **Validation** (`validate_configs`):
   - Checks configuration validity
   - Verifies file paths exist
   - Ensures required parameters are set

---

### 5. `dataset_utils.py`
**Purpose**: Dataset loading, preprocessing, and augmentation

**Key Classes**:

#### `Dataset` Class
Main dataset class inheriting from `torch.utils.data.Dataset`

**Methods**:

1. **`__init__`**: 
   - Loads pickle files with split information
   - Extracts metadata (bands, means, stds)
   - Configures augmentation settings

2. **`load_img`**:
   - Loads images from disk (NumPy or PyTorch format)
   - Handles multi-temporal data (before/after images)
   - Loads masks (cloud, sea, land cover) if needed

3. **`scale_img`**:
   - Normalizes images using dataset statistics
   - Multiple scaling options:
     - `"normalize"`: Z-score normalization
     - `"min-max"`: Min-max scaling to [0,1]
     - `"clamp_scale"`: Clamp then scale
     - Custom range scaling

4. **`fillna`**:
   - Replaces NaN values with configured constant
   - Marks corresponding label pixels as `2` (ignored)

5. **`augment`**:
   - Random horizontal flip (p=0.5)
   - Random vertical flip (p=0.5)
   - Random rotation (-15° to +15°)

6. **`__getitem__`**:
   - Complete pipeline: load → fillna → scale → augment
   - Returns dictionary with images and metadata

#### `OverSampler` Class
Custom sampler for handling class imbalance

**How it works**:
- Oversamples positive (burnt) patches
- Configurable positive:negative ratio
- Random selection with replacement for positives
- Balances batch composition during training

---

### 6. `cd_experiments_utils.py`
**Purpose**: Training and evaluation loops for change detection

**Key Functions**:

#### `train_change_detection()`
Complete training loop for one repetition

**Process**:
1. Initialize metrics (confusion matrix, IoU)
2. Setup loss function, optimizer, scheduler
3. For each epoch:
   - **Training phase**:
     - Forward pass through model
     - Calculate loss
     - Backward pass with gradient descent
     - Update learning rate
     - Log metrics
   - **Validation phase**:
     - Evaluate on validation set
     - Save checkpoint if best F1-score
     - Visualize predictions on sample image
4. Log results to WandB (if enabled)

**Features**:
- Mixed precision training support
- Multi-scale inference (for ChangeFormer)
- Checkpoint saving strategies
- Progress bars with live metrics

#### `eval_change_detection()`
Evaluation loop for validation/test sets

**Process**:
1. Set model to evaluation mode
2. For each batch:
   - Forward pass (no gradients)
   - Calculate predictions
   - Compute metrics per sample
3. Aggregate metrics:
   - Accuracy, Precision, Recall, F1-score, IoU
   - Overall metrics
   - Land cover-specific metrics (if enabled)
4. Visualize sample predictions
5. Return comprehensive results

**Metrics Computed**:
- Per-class: Accuracy, Precision, Recall, F1, IoU
- Overall: Mean F1, Mean IoU
- Land cover breakdown (optional)

---

## 🤖 Models & Architectures

The project implements **10 state-of-the-art change detection models**:

### 1. **BAM-CD** (Burnt Area Mapping - Change Detection)
**The proposed model in the paper**

**Architecture**:
- **Encoder**: ResNet-based backbone (configurable depth)
- **Fusion**: Concatenation or difference of bi-temporal features
- **Decoder**: U-Net style decoder with skip connections
- **Attention**: SCSE (Spatial and Channel Squeeze & Excitation)
- **Mode**: Siamese (shared encoder) or pseudo-siamese

**Versions**:
- **V1**: Siamese architecture with SCSE attention
- **V2**: Pseudo-siamese with enhanced data augmentation

**Configuration** (`bam_cd.json`):
```json
{
    "backbone": "resnet101",
    "siamese": true,
    "decoder_attention_type": "scse",
    "activation": null
}
```

### 2. **U-Net**
Classic U-Net architecture adapted for change detection

**Features**:
- Concatenates before/after images as input
- Encoder-decoder with skip connections
- Dropout regularization (p=0.2)
- 4 downsampling levels

### 3. **FC-EF** (Fully Convolutional Early Fusion)
Two variants:
- **FC-EF-Conc**: Concatenation fusion
- **FC-EF-Diff**: Difference fusion

**Architecture**:
- Early fusion of temporal images
- Fully convolutional design
- Simple and fast baseline

### 4. **SNUNet**
**SNUNet-ECAM** (Efficient Channel Attention Module)

**Features**:
- Dense skip connections
- Channel attention mechanism
- Lightweight architecture

### 5. **HFANet**
**Hierarchical Feature Aggregation Network**

**Features**:
- Multi-scale feature extraction
- Hierarchical attention
- Complex feature fusion

### 6. **ChangeFormer**
**Transformer-based change detection**

**Features**:
- Vision transformer encoder
- Multi-scale prediction
- Self-attention mechanisms
- Multiple output heads

**Special Training**:
- Multi-scale training with weighted loss
- Multi-scale inference at test time

### 7. **BIT-CD**
**Binary Image Transformer for Change Detection**

**Features**:
- Transformer-based architecture
- Binary change detection focus
- Context-aware attention

### 8. **ADHR-CDNet**
**Attentive Deformable Hierarchical Recurrent Network**

**Features**:
- Deformable convolutions
- Recurrent connections
- Multi-level attention

### 9. **TransUNet-CD**
**TransUNet adapted for Change Detection**

**Features**:
- Hybrid CNN-Transformer
- U-Net structure with transformer blocks
- Patch-based processing

### 10. **Various Baselines**
Additional baseline models for comparison

---

## 🔄 Data Pipeline

### Complete Data Flow:

```
1. RAW DATA (HDF5 files)
   ↓
   [create_dataset.py - Stage 1]
   ↓
2. CROPPED PATCHES (.npy or .pt files)
   - Before images (Sentinel-2/MODIS)
   - After images (Sentinel-2/MODIS)
   - Labels
   - Masks (cloud, sea, land cover)
   ↓
   [create_dataset.py - Stage 2]
   ↓
3. SPLIT METADATA (.pkl files)
   - train.pkl: Training patch paths + metadata
   - val.pkl: Validation patch paths + metadata
   - test.pkl: Test patch paths + metadata
   ↓
   [Dataset class - __getitem__]
   ↓
4. RUNTIME PROCESSING
   - Load images from disk
   - Fill NaN values
   - Normalize/scale images
   - Apply augmentation (training only)
   ↓
5. BATCHED TENSORS
   - Shape: (batch_size, channels, height, width)
   - Ready for model input
   ↓
   [Model forward pass]
   ↓
6. PREDICTIONS
   - Shape: (batch_size, num_classes, height, width)
   - Logits or probabilities
```

### Data Preprocessing Details:

**Scaling Options** (`scale_input` in config):

1. **`"normalize"`**: Z-score normalization
   ```
   x_norm = (x - mean) / std
   ```

2. **`"min-max"`**: Min-max scaling to [0, 1]
   ```
   x_scaled = (x - min) / (max - min)
   ```

3. **`"clamp_scale_10000"`**: Clamp then scale
   ```
   x_clamped = min(x, 10000)
   x_scaled = x_clamped / 10000
   ```

4. **Custom range**: `[new_min, new_max]`
   ```
   x_scaled = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
   ```

**NaN Handling**:
- NaN pixels replaced with `nan_value` (default: 0)
- Corresponding label pixels marked as `2` (ignored in loss)

**Augmentation** (training only):
- Horizontal flip: 50% probability
- Vertical flip: 50% probability
- Rotation: 50% probability, angle ∈ [-15°, +15°]

---

## 🎓 Training & Evaluation

### Training Process:

**1. Initialization**:
```python
# Load configurations
configs = load(config.json)
model_configs = load(method_config.json)

# Initialize model
model = init_model(configs, model_configs, ...)

# Initialize optimizer
optimizer = Adam(lr=0.001, weight_decay=0.0)

# Initialize loss
criterion = CrossEntropyLoss(weights=[w0, w1])
```

**2. Training Loop**:
```python
for epoch in range(n_epochs):
    # Training phase
    for batch in train_loader:
        before, after, label = batch
        
        # Forward pass
        output = model(before, after)
        loss = criterion(output, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log metrics
        compute_metrics(output, label)
    
    # Validation phase
    val_metrics = evaluate(model, val_loader)
    
    # Save best model
    if val_metrics['f1'] > best_f1:
        save_checkpoint(model, epoch)
        best_f1 = val_metrics['f1']
    
    # Update learning rate
    lr_scheduler.step()
```

**3. Evaluation**:
```python
# Load best checkpoint
checkpoint = load(best_checkpoint.pt)
model.load_state_dict(checkpoint)

# Evaluate on test set
results = evaluate(model, test_loader)

# Compute final metrics
print(f"F1-score: {results['f1']}")
print(f"IoU: {results['iou']}")
```

### Metrics:

**Per-class Metrics** (burnt and unburnt):
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-score**: 2 × (Precision × Recall) / (Precision + Recall)
- **IoU**: TP / (TP + FP + FN)

**Aggregate Metrics**:
- **Mean F1**: Average of F1 scores for both classes
- **Mean IoU**: Average of IoU for both classes

**Land Cover Metrics** (optional):
- All above metrics broken down by Corine Land Cover class
- Useful for understanding model performance on different terrain types

### Checkpoint Strategy:

**Saving Options** (`save_checkpoint_freq`):
- `-1`: Save only the best model (based on validation F1)
- `N`: Save every N epochs
- `[N, M]`: Save every N epochs until epoch M, then save every epoch

**Checkpoint Contents**:
```python
{
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': scheduler.state_dict(),
    'loss': current_loss
}
```

---

## ⚙️ Configuration System

### Main Configuration (`configs/config.json`)

**Key Sections**:

1. **Method Selection**:
```json
{
    "method": "bam_cd",  // Which model to use
    "mode": "train",     // train or eval
}
```

2. **Dataset Configuration**:
```json
{
    "dataset_type": "sen2_20_mod_500",  // sen2_<gsd>_mod_<gsd>
    "datasets": {
        "data_source": "sen2",  // sen2 or mod
        "img_size": 256,
        "batch_size": 8,
        "augmentation": false,
        "scale_input": "clamp_scale_10000"
    }
}
```

3. **Training Configuration**:
```json
{
    "train": {
        "n_epochs": 150,
        "rep_times": 10,        // Repeat experiment 10 times
        "loss_function": "cross_entropy",
        "weighted_loss": false,
        "mixed_precision": false
    }
}
```

4. **Paths**:
```json
{
    "paths": {
        "dataset": "/path/to/dataset",
        "results": "results/",
        "load_state": null      // Path to checkpoint
    }
}
```

5. **Band Selection**:
```json
{
    "selected_bands": {
        "sen2": {
            "B02": -1,  // Blue
            "B03": -1,  // Green
            "B04": -1,  // Red
            "B8A": -1   // NIR
            // -1 means index will be filled at runtime
        }
    }
}
```

### Model Configuration (`configs/method/*.json`)

Each model has its own configuration file:

**Example** (`bam_cd.json`):
```json
{
    "optimizer": {
        "name": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "lr_schedule": null
    },
    "backbone": "resnet101",
    "encoder_weights": null,
    "siamese": true,
    "decoder_attention_type": "scse"
}
```

**Common Parameters**:
- `optimizer`: Optimizer configuration
- `learning_rate`: Initial learning rate
- `lr_schedule`: Learning rate scheduling strategy
- Model-specific architecture parameters

---

## 📊 Loss Functions

Located in `losses/` directory:

### 1. **Cross-Entropy Loss**
Standard classification loss with class weights

**Formula**:
```
L = -Σ w_c × y_c × log(p_c)
```
- `w_c`: Class weight
- `y_c`: Ground truth (one-hot)
- `p_c`: Predicted probability

**Use case**: Balanced datasets or with class weighting

### 2. **Dice Loss** (`losses/dice.py`)
Segmentation-specific loss based on Dice coefficient

**Formula**:
```
Dice = (2 × |X ∩ Y|) / (|X| + |Y|)
L_dice = 1 - Dice
```

**Features**:
- Handles class imbalance naturally
- Smooth gradients
- Directly optimizes IoU-like metric

**Use case**: Highly imbalanced segmentation tasks

### 3. **Combined BCE + Dice Loss** (`losses/bce_and_dice.py`)
Combines benefits of both losses

**Formula**:
```
L = L_CE + L_Dice
```

**Features**:
- CE provides class discrimination
- Dice handles spatial overlap
- Best of both worlds

**Use case**: General segmentation tasks, especially with imbalance

### 4. **Focal Loss**
Addresses extreme class imbalance

**Formula**:
```
L_focal = -α × (1-p)^γ × log(p)
```
- `α`: Class weight
- `γ`: Focusing parameter (default: 2)

**Use case**: Extremely imbalanced datasets

---

## 🚀 Usage Guide

### Step-by-Step Workflow:

#### **1. Download FLOGA Dataset**
```bash
# Download from Dropbox link in README
# Place .h5 files in a directory, e.g., data/raw/
```

#### **2. Create Analysis-Ready Dataset**
```bash
python create_dataset.py \
    --floga_path data/raw/ \
    --out_path data/processed/ \
    --out_size 256 256 \
    --stages 1 2 \
    --sample 1 \
    --ratio 60 20 20 \
    --random_seed 999
```

**Output**:
- Cropped patches in `data/processed/2017/`, `data/processed/2018/`, etc.
- Split metadata: `train.pkl`, `val.pkl`, `test.pkl`

#### **3. Configure Experiment**
Edit `configs/config.json`:
```json
{
    "method": "bam_cd",
    "mode": "train",
    "dataset_type": "sen2_20_mod_500",
    "paths": {
        "dataset": "data/processed/",
        "results": "results/"
    },
    "datasets": {
        "data_source": "sen2",
        "batch_size": 8
    },
    "train": {
        "n_epochs": 150,
        "rep_times": 1
    }
}
```

#### **4. Train Model**
```bash
python run_experiment.py --config configs/config.json
```

**What happens**:
- Model trains for 150 epochs
- Validates after each epoch
- Saves best checkpoint to `results/bam_cd/<timestamp>/checkpoints/`
- Logs metrics (console and/or WandB)

#### **5. Evaluate Model**
Edit `configs/config.json`:
```json
{
    "mode": "eval",
    "paths": {
        "load_state": "results/bam_cd/<timestamp>/checkpoints/0/best_segmentation.pt"
    }
}
```

Run evaluation:
```bash
python run_experiment.py --config configs/config.json
```

#### **6. Visualize Predictions**
```bash
python visualize_predictions.py \
    --results_path results/visualizations/ \
    --config configs/config.json \
    --bands nrg \
    --mode test \
    --export_numpy_preds
```

**Output**:
- Visualizations in `results/visualizations/test/`
- Separate folders for TP, FP, FN cases
- NumPy arrays of predictions

---

### Advanced Usage:

#### **Multi-Repetition Experiments**
Run same experiment multiple times for statistical significance:
```json
{
    "train": {
        "rep_times": 10
    }
}
```

Results averaged across all repetitions.

#### **Resume Training**
```json
{
    "train": {
        "resume": true
    },
    "paths": {
        "load_state": "path/to/checkpoint.pt"
    }
}
```

#### **Mixed Precision Training**
Faster training with reduced memory:
```json
{
    "train": {
        "mixed_precision": true
    }
}
```

#### **WandB Logging**
```json
{
    "wandb": {
        "activate": true,
        "wandb_project": "floga-experiments",
        "wandb_entity": "your-username"
    }
}
```

#### **Class Weighting**
Handle class imbalance:
```json
{
    "train": {
        "weighted_loss": true
    }
}
```

Weights automatically computed from dataset statistics.

#### **Land Cover Analysis**
Get per-land-cover-type metrics:
```json
{
    "train": {
        "log_landcover_metrics": true
    }
}
```

Results saved to `lc_stats.csv`.

---

## 🔍 Understanding the Code Flow

### **Training Flow**:
```
run_experiment.py
    ├─> Load configs
    ├─> Initialize Dataset (dataset_utils.py)
    │   └─> Load pickle metadata
    │   └─> Define augmentations
    ├─> Initialize Model (utils.py: init_model)
    │   └─> Factory creates model instance
    │   └─> Load pretrained weights if available
    ├─> Initialize Optimizer & Scheduler (utils.py)
    ├─> Initialize Loss Function (utils.py: create_loss)
    ├─> Train Loop (cd_experiments_utils.py: train_change_detection)
    │   ├─> For each epoch:
    │   │   ├─> Training phase
    │   │   │   └─> Forward → Loss → Backward → Step
    │   │   └─> Validation phase
    │   │       └─> Evaluate → Save if best
    │   └─> Return trained model
    └─> Final Evaluation (cd_experiments_utils.py: eval_change_detection)
        └─> Test set → Metrics → Save results
```

### **Data Loading Flow**:
```
DataLoader
    └─> Dataset.__getitem__(idx)
        ├─> Load images from disk (load_img)
        │   └─> Before/After images + Label + Masks
        ├─> Fill NaN values (fillna)
        │   └─> NaN → nan_value, Label → 2
        ├─> Scale images (scale_img)
        │   └─> Normalization or min-max scaling
        └─> Augment (if training) (augment)
            └─> Flips + Rotation
        
        Returns: Dict with tensors ready for model
```

---

## 📈 Key Insights from Code

### **1. Handling Temporal Data**:
Models receive two inputs:
- `before_img`: Pre-fire image
- `after_img`: Post-fire image

Different fusion strategies:
- **Early fusion**: Concatenate inputs (U-Net, FC-EF-Conc)
- **Late fusion**: Process separately, fuse features (BAM-CD siamese)
- **Difference**: Compute difference (FC-EF-Diff)

### **2. Label Handling**:
Three label values:
- `0`: Unburnt (negative class)
- `1`: Burnt (positive class)
- `2`: Ignore (burnt in other events or NaN)

All loss functions configured with `ignore_index=2`

### **3. Class Imbalance**:
Multiple strategies employed:
- **Class weighting**: Automatic weight calculation
- **Oversampling**: `OverSampler` class
- **Focal loss**: Alternative loss function
- **Dice loss**: Handles imbalance inherently

### **4. Evaluation Strategy**:
- **Best model selection**: Based on validation F1-score for burnt class
- **Multiple repetitions**: Statistical robustness
- **Comprehensive metrics**: Per-class and aggregate
- **Land cover analysis**: Terrain-specific performance

### **5. Reproducibility**:
- Fixed random seeds (999)
- Deterministic splits in dataset creation
- Checkpoint all hyperparameters
- WandB logging for experiment tracking

---

## 🎯 Summary

This codebase is a complete framework for:
1. **Dataset creation** from raw satellite data
2. **Model training** with multiple state-of-the-art architectures
3. **Comprehensive evaluation** with various metrics
4. **Visualization** and result analysis

**Key Strengths**:
- Modular design (easy to add new models)
- Extensive configuration options
- Robust data pipeline
- Multiple baseline models
- Comprehensive evaluation metrics

**Use Cases**:
- Wildfire burnt area mapping
- Change detection research
- Satellite image segmentation
- Benchmark comparison of CD models

---

## 📚 References

For citation and more details, see the paper:
```
Sdraka et al. (2024). "FLOGA: A Machine-Learning-Ready Dataset, a Benchmark, 
and a Novel Deep Learning Model for Burnt Area Mapping With Sentinel-2"
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
```

---

*Generated on: January 15, 2026*
