# 🔥 Wildfire Impact Detection: Complete ML Interview Guide

> A comprehensive guide to understand all stages, ML concepts, and terminology in this project for interviews.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [ML Fundamentals Used](#2-ml-fundamentals-used)
3. [Complete Pipeline Stages](#3-complete-pipeline-stages)
4. [Deep Learning Architecture Terms](#4-deep-learning-architecture-terms)
5. [Training Process Deep Dive](#5-training-process-deep-dive)
6. [Evaluation Metrics Explained](#6-evaluation-metrics-explained)
7. [Data Pipeline & Preprocessing](#7-data-pipeline--preprocessing)
8. [Model Architectures in This Project](#8-model-architectures-in-this-project)
9. [Loss Functions Explained](#9-loss-functions-explained)
10. [Interview Questions & Answers](#10-interview-questions--answers)
11. [Common Terms Glossary](#11-common-terms-glossary)

---

## 1. Project Overview

### What This Project Does

This is a **Change Detection** project for **Semantic Segmentation** of satellite imagery. It detects:
- **Wildfire impacts**: Areas burnt by wildfires
- **Drought impacts**: Areas affected by drought conditions

### The Core ML Task

**Input**: Two satellite images (before and after an event)
**Output**: A binary mask showing which pixels changed (burnt/drought-affected)

```
Before Image (T1) ─┐
                   ├─→ [Neural Network] ─→ Change Mask (0=unchanged, 1=changed)
After Image (T2)  ─┘
```

### Why This is Important for ML Interviews

1. **Computer Vision**: Image segmentation, CNNs, encoder-decoder architectures
2. **Deep Learning**: Training loops, loss functions, optimization
3. **Remote Sensing**: Real-world satellite data challenges
4. **Change Detection**: Temporal analysis, bitemporal learning

---

## 2. ML Fundamentals Used

### Supervised Learning
```
Training Data = (X, Y) pairs where:
  X = Satellite image pairs (before, after)
  Y = Ground truth labels (burnt/unburnt mask)
```

### Semantic Segmentation
- **Definition**: Classifying each pixel in an image
- **Output**: A mask with same spatial dimensions as input
- **In this project**: Binary segmentation (2 classes: changed vs unchanged)

### Convolutional Neural Networks (CNNs)
```
Why CNNs for Images?
├── Spatial hierarchy: Learn from local to global patterns
├── Parameter sharing: Same filter applied across image
├── Translation invariance: Detect features anywhere in image
└── Hierarchical features: Edges → Textures → Objects
```

### Encoder-Decoder Architecture
```
       ENCODER                    DECODER
    (Downsampling)              (Upsampling)
    
Input ──┬── 256×256        16×16 ──┬── 256×256 Output
        │   ↓                 ↑    │
        ├── 128×128        32×32 ──┤
        │   ↓                 ↑    │
        ├── 64×64          64×64 ──┤  ← Skip Connections
        │   ↓                 ↑    │
        ├── 32×32         128×128 ─┤
        │   ↓                 ↑    │
        └── 16×16 ─────────────────┘
        (Bottleneck / Latent Space)
```

**Encoder**: Extracts features, reduces spatial dimensions
**Decoder**: Reconstructs spatial resolution from features
**Skip Connections**: Preserve fine details lost during encoding

---

## 3. Complete Pipeline Stages

### Stage 1: Data Preparation (`create_dataset.py`)

```
Raw HDF5 Files (Satellite Data)
         │
         ▼
┌─────────────────────────┐
│  1. Load raw imagery    │ ← HDF5 compressed files with Sentinel-2/MODIS data
│  2. Crop into patches   │ ← 256×256 pixel patches
│  3. Filter bad data     │ ← Remove cloudy, sea-covered patches
│  4. Train/Val/Test split│ ← Typically 70/15/15 or 80/10/10
│  5. Save as .npy files  │ ← NumPy arrays for fast loading
└─────────────────────────┘
         │
         ▼
Processed Dataset (.npy patches + .pkl metadata)
```

**Interview Point**: Data preprocessing is crucial. 80% of ML work is data preparation!

### Stage 2: Data Loading (`dataset_utils.py`)

```python
# PyTorch Dataset class - called for each training sample
class Dataset:
    def __getitem__(self, idx):
        # 1. Load images from disk
        before_img = load_npy(before_path)
        after_img = load_npy(after_path)
        label = load_npy(label_path)
        
        # 2. Handle missing values (NaN handling)
        before_img = fill_nan(before_img)
        
        # 3. Normalize/Scale images
        before_img = normalize(before_img, mean, std)
        
        # 4. Apply augmentations (training only)
        if self.mode == 'train':
            before_img, after_img, label = augment(before_img, after_img, label)
        
        return {'before': before_img, 'after': after_img, 'label': label}
```

**Key Concepts**:
- **DataLoader**: Batches data, handles shuffling, parallel loading
- **Batch Size**: Number of samples processed together
- **Data Augmentation**: Artificially increase training data diversity

### Stage 3: Model Initialization (`utils.py`)

```python
# Factory pattern to create different models
def init_model(configs, model_configs, checkpoint, num_channels, device):
    if configs['method'] == 'bam_cd':
        model = BAM_CD(encoder_name='resnet34', in_channels=num_channels, classes=2)
    elif configs['method'] == 'unet':
        model = Unet(in_channels=num_channels, classes=2)
    # ... more models
    
    # Load pretrained weights if available
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)  # Move to GPU
    return model
```

### Stage 4: Training Loop (`cd_experiments_utils.py`)

```python
for epoch in range(n_epochs):
    model.train()  # Set to training mode (enables dropout, batch norm updates)
    
    for batch in train_loader:
        # ===== FORWARD PASS =====
        before_img = batch['before'].to(device)  # Shape: (B, C, H, W)
        after_img = batch['after'].to(device)    # Shape: (B, C, H, W)
        label = batch['label'].to(device)        # Shape: (B, H, W)
        
        # Model predicts change probability for each pixel
        output = model(before_img, after_img)    # Shape: (B, 2, H, W)
        
        # ===== COMPUTE LOSS =====
        loss = criterion(output, label)          # Scalar value
        
        # ===== BACKWARD PASS =====
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients via backpropagation
        optimizer.step()       # Update weights
    
    # ===== VALIDATION =====
    model.eval()  # Set to evaluation mode (disables dropout)
    with torch.no_grad():  # No gradient computation needed
        val_metrics = evaluate(model, val_loader)
    
    # ===== CHECKPOINT =====
    if val_metrics['f1'] > best_f1:
        save_checkpoint(model, epoch)
        best_f1 = val_metrics['f1']
```

### Stage 5: Evaluation (`cd_experiments_utils.py`)

```python
def eval_change_detection(model, test_loader, device):
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch['before'], batch['after'])
            predictions = output.argmax(dim=1)  # Get class with highest probability
            
            all_predictions.append(predictions)
            all_labels.append(batch['label'])
    
    # Compute metrics
    metrics = {
        'accuracy': compute_accuracy(predictions, labels),
        'precision': compute_precision(predictions, labels),
        'recall': compute_recall(predictions, labels),
        'f1_score': compute_f1(predictions, labels),
        'iou': compute_iou(predictions, labels)
    }
    
    return metrics
```

### Stage 6: Visualization (`visualize_predictions.py`)

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Before Image   │  After Image    │  Ground Truth   │  Prediction     │
│  (RGB/NRG)      │  (RGB/NRG)      │  (Binary Mask)  │  (Binary Mask)  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## 4. Deep Learning Architecture Terms

### Encoder (Backbone/Feature Extractor)

```
Purpose: Extract hierarchical features from input images

Input: Raw image (B, C, H, W)
       ↓
Conv Block 1 → Features (B, 64, H/2, W/2)
       ↓
Conv Block 2 → Features (B, 128, H/4, W/4)
       ↓
Conv Block 3 → Features (B, 256, H/8, W/8)
       ↓
Conv Block 4 → Features (B, 512, H/16, W/16)
       ↓
Conv Block 5 → Features (B, 512, H/32, W/32)  ← Bottleneck (most abstract features)
```

**Common Encoders Used**:
| Encoder | Parameters | Use Case |
|---------|------------|----------|
| ResNet-34 | 21M | Good balance of speed/accuracy |
| ResNet-50 | 25M | Better accuracy, more compute |
| EfficientNet | Varies | State-of-the-art efficiency |

### Decoder (Upsampling Path)

```
Purpose: Recover spatial resolution while maintaining learned features

Bottleneck (B, 512, H/32, W/32)
       ↓
Upsample + Conv → (B, 256, H/16, W/16)
       ↓
Upsample + Conv → (B, 128, H/8, W/8)
       ↓
Upsample + Conv → (B, 64, H/4, W/4)
       ↓
Upsample + Conv → (B, 32, H/2, W/2)
       ↓
Final Conv → (B, num_classes, H, W)  ← Output: per-pixel class probabilities
```

### Skip Connections

```
Why Skip Connections?
├── Problem: Deep networks lose fine spatial details
├── Solution: Connect encoder layers directly to decoder layers
└── Result: Combine high-level semantics + low-level details

Encoder Layer 3 ──────────────────────────┐
(Fine details)                            │
                                          ▼
Encoder Layer 4 ────────────────┐    [Concatenate]
                                │         │
                                ▼         ▼
Encoder Layer 5 ──→ Decoder 5 ──→ Decoder 4 ──→ Decoder 3
(Abstract features)
```

### Siamese Networks (Used in Change Detection)

```
Before Image ──→ [Encoder] ──→ Features_before
                    │                     │
                    │ (Shared             │ [Fusion: Concat/Diff]
                    │  Weights)           │
                    │                     ▼
After Image ───→ [Encoder] ──→ Features_after ──→ [Decoder] ──→ Change Mask
```

**Key Point**: Same encoder processes both images, ensuring fair comparison.

### Batch Normalization

```python
# Normalizes activations across a batch
# Helps with:
# 1. Training stability
# 2. Faster convergence
# 3. Acts as regularization

BatchNorm(x) = γ * (x - μ_batch) / σ_batch + β

Where:
- μ_batch, σ_batch: mean and std of current batch
- γ, β: learnable parameters (scale and shift)
```

### Dropout

```python
# Randomly zeros some neurons during training
# Helps prevent overfitting

During Training: randomly drop 20-50% of neurons
During Evaluation: use all neurons (dropout disabled)

# Critical Interview Point:
model.train()  # Dropout ON
model.eval()   # Dropout OFF - MUST call before evaluation!
```

### Activation Functions

```
ReLU: f(x) = max(0, x)
├── Most common activation
├── Fast computation
└── Problem: "dying ReLU" (negative inputs always 0)

Sigmoid: f(x) = 1 / (1 + e^(-x))
├── Output range: [0, 1]
├── Used for binary classification final layer
└── Problem: vanishing gradients

Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j))
├── Outputs sum to 1 (probability distribution)
├── Used for multi-class classification
└── In this project: final layer outputs class probabilities
```

---

## 5. Training Process Deep Dive

### Optimization

#### Gradient Descent

```
Goal: Minimize loss function by adjusting weights

Algorithm:
1. Forward pass: compute predictions
2. Compute loss: measure prediction error
3. Backward pass: compute gradients (∂Loss/∂weights)
4. Update weights: w_new = w_old - learning_rate × gradient
```

#### Optimizers

```python
# SGD (Stochastic Gradient Descent)
optimizer = SGD(lr=0.01, momentum=0.9)
# Simple but may get stuck in local minima

# Adam (Adaptive Moment Estimation) - Most popular
optimizer = Adam(lr=0.001, betas=(0.9, 0.999))
# Combines momentum + adaptive learning rates per parameter

# AdamW (Adam with Weight Decay)
optimizer = AdamW(lr=0.001, weight_decay=0.01)
# Better generalization through proper weight decay
```

#### Learning Rate Scheduling

```python
# Cosine Annealing - gradually decrease LR
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Step Decay - decrease by factor every N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# ReduceLROnPlateau - decrease when validation plateaus
scheduler = ReduceLROnPlateau(optimizer, patience=10)
```

**Why Schedule LR?**
- High LR at start: explore parameter space quickly
- Low LR at end: fine-tune to precise minimum

### Backpropagation

```
Forward Pass:
Input ──→ Layer1 ──→ Layer2 ──→ Layer3 ──→ Output ──→ Loss

Backward Pass (Chain Rule):
∂L/∂w1 = ∂L/∂Output × ∂Output/∂Layer3 × ∂Layer3/∂Layer2 × ∂Layer2/∂Layer1 × ∂Layer1/∂w1

PyTorch handles this automatically:
loss.backward()  # Computes all gradients
optimizer.step() # Updates all weights
```

### Mixed Precision Training (AMP)

```python
# Uses FP16 (16-bit) instead of FP32 (32-bit) where safe
# Benefits: 2x faster, 50% less memory

scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Overfitting vs Underfitting

```
Underfitting                      Optimal                         Overfitting
High train loss                   Low train loss                  Very low train loss
High val loss                     Low val loss                    High val loss
│                                 │                               │
└── Model too simple              └── Good generalization         └── Memorized training data
    Add capacity                      Keep this!                      Add regularization
```

### Regularization Techniques

```python
# 1. L2 Regularization (Weight Decay)
optimizer = Adam(lr=0.001, weight_decay=0.01)
# Penalizes large weights: Loss_total = Loss + λ × Σ(w²)

# 2. Dropout
nn.Dropout(p=0.5)  # 50% of neurons dropped randomly

# 3. Data Augmentation
# Artificially increase dataset diversity
transforms = [RandomFlip(), RandomRotation(), ColorJitter()]

# 4. Early Stopping
if val_loss not improving for N epochs:
    stop_training()
```

---

## 6. Evaluation Metrics Explained

### Confusion Matrix

```
                    Predicted
                 Neg    |    Pos
Actual   Neg     TN     |    FP     ← False Positive (Type I Error)
         Pos     FN     |    TP     ← True Positive
                 ↑
                 False Negative (Type II Error)

TN: True Negative  - Correctly predicted unchanged
TP: True Positive  - Correctly predicted changed
FP: False Positive - Predicted changed, actually unchanged (False alarm)
FN: False Negative - Predicted unchanged, actually changed (Missed detection)
```

### Key Metrics

```python
# ACCURACY: Overall correctness
Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Problem: Misleading for imbalanced classes!
# If 95% pixels unchanged, predicting all unchanged = 95% accuracy

# PRECISION: Of predicted positives, how many correct?
Precision = TP / (TP + FP)
# "When model says burnt, how often is it right?"
# High precision = few false alarms

# RECALL (Sensitivity): Of actual positives, how many detected?
Recall = TP / (TP + FN)
# "Of all burnt areas, how many did we find?"
# High recall = few missed detections

# F1-SCORE: Harmonic mean of precision and recall
F1 = 2 × (Precision × Recall) / (Precision + Recall)
# Balanced metric when both precision and recall matter
# Range: [0, 1], higher is better

# IoU (Intersection over Union) / Jaccard Index
IoU = TP / (TP + FP + FN)
# Standard metric for segmentation
# Measures overlap between prediction and ground truth
# Range: [0, 1], higher is better
```

### Visual Explanation of IoU

```
Ground Truth:       Prediction:         Intersection:       Union:
██████████         ████████████        ████████            ████████████
██████████         ████████████        ████████            ████████████
██████████         ████████████        ████████            ████████████
██████████                             ████                ██████████
██████████                                                 ██████████

IoU = |Intersection| / |Union| = 20 / 30 = 0.67
```

### Why IoU and F1 for Segmentation?

```
1. Class Imbalance Problem:
   - Burnt areas often < 5% of image
   - Accuracy would be misleading (95%+ just by predicting all unburnt)

2. IoU is Stricter:
   - Penalizes both false positives AND false negatives
   - Standard metric for segmentation competitions

3. F1 for Balanced View:
   - Precision: Don't cry wolf (few false alarms)
   - Recall: Don't miss fires (few misses)
   - F1 balances both concerns
```

---

## 7. Data Pipeline & Preprocessing

### Satellite Image Bands (Sentinel-2)

```
Band    Name            Wavelength    What it Detects
B2      Blue            490 nm        Water, atmosphere
B3      Green           560 nm        Vegetation health
B4      Red             665 nm        Chlorophyll absorption
B5      Red Edge 1      705 nm        Vegetation stress
B6      Red Edge 2      740 nm        Vegetation stress  
B7      Red Edge 3      783 nm        Vegetation structure
B8      NIR             842 nm        Biomass, water content
B8A     Narrow NIR      865 nm        Vegetation/water
B11     SWIR 1          1610 nm       Fire detection, soil
B12     SWIR 2          2190 nm       Fire detection, burnt areas

Why Multiple Bands?
├── RGB (B4, B3, B2): Human-visible imagery
├── NIR (B8): Healthy vegetation reflects strongly
├── SWIR (B11, B12): Fire scars absorb strongly
└── Red Edge: Sensitive to vegetation stress
```

### Data Normalization

```python
# Z-Score Normalization (Most common)
normalized = (x - mean) / std
# Mean=0, Std=1 after normalization
# Used when: different bands have different scales

# Min-Max Scaling
scaled = (x - min) / (max - min)
# Range becomes [0, 1]
# Used when: bounded range needed

# Why Normalize?
# 1. Neural networks train better with similar-scale inputs
# 2. Prevents large values from dominating gradients
# 3. Helps optimization converge faster
```

### Data Augmentation

```python
# Spatial augmentations (applied to both images + label)
transforms = [
    RandomHorizontalFlip(p=0.5),     # Flip left-right
    RandomVerticalFlip(p=0.5),       # Flip up-down
    RandomRotation(degrees=90),      # Rotate 0°, 90°, 180°, 270°
    RandomCrop(size=224),            # Random crop (if images larger)
]

# Why Augment?
# 1. Increases effective dataset size
# 2. Makes model invariant to transformations
# 3. Reduces overfitting

# Important for Change Detection:
# Same augmentation must be applied to BOTH before AND after images!
```

### Handling Missing Data

```python
# Satellite images often have:
# - Cloud-covered pixels (invalid)
# - Sensor errors (NaN values)

# Solution 1: Fill with mean/median
image[np.isnan(image)] = band_mean

# Solution 2: Mask out invalid pixels
loss = criterion(pred[valid_mask], label[valid_mask])

# This project uses ignore_index=2 in loss
# Pixels labeled as 2 (other events/uncertain) are ignored
```

### Train/Validation/Test Split

```
Full Dataset
     │
     ├── 70% Training Set
     │   └── Used to train model weights
     │
     ├── 15% Validation Set
     │   └── Used to tune hyperparameters
     │   └── Monitor overfitting
     │   └── Save best checkpoint
     │
     └── 15% Test Set
         └── Final evaluation only
         └── Never seen during training!
         └── Measures real-world performance
```

### Class Imbalance Handling

```python
# Problem: Burnt areas << Unburnt areas
# Example: 5% burnt, 95% unburnt

# Solution 1: Class Weights
class_weights = compute_class_weights(dataset)
# weight_burnt = total_pixels / (2 × burnt_pixels)
# Higher weight = model penalized more for missing this class
criterion = CrossEntropyLoss(weight=class_weights)

# Solution 2: Oversampling
# Include more positive samples during training
sampler = OverSampler(dataset, positive_prc=0.5)
# Now 50% of batches contain burnt pixels

# Solution 3: Focal Loss
# Automatically down-weight easy examples
FocalLoss = -α(1-p)^γ × log(p)
```

---

## 8. Model Architectures in This Project

### 1. U-Net (`models/unet.py`)

```
Classic encoder-decoder for segmentation

Architecture:
Input ──→ [Encoder: VGG/ResNet] ──→ [Decoder with Skip Connections] ──→ Mask

Key Features:
├── Skip connections preserve spatial details
├── Works well with limited data
└── Standard baseline for medical/satellite segmentation
```

### 2. BAM-CD (`models/bam_cd/`)

```
Proposed model in the FLOGA paper

Architecture:
Before ──→ [Siamese Encoder] ──┐
                               ├──→ [Feature Fusion] ──→ [Decoder] ──→ Change Mask  
After ───→ [Siamese Encoder] ──┘

Key Features:
├── Siamese: Same encoder for both images (fair comparison)
├── Fusion modes: Concatenation or Difference
├── ResNet backbone: Pretrained on ImageNet
└── State-of-the-art on FLOGA dataset
```

### 3. ChangeFormer (`models/changeformer.py`)

```
Transformer-based change detection

Architecture:
Input ──→ [CNN Encoder] ──→ [Transformer Blocks] ──→ [Decoder] ──→ Mask

Key Features:
├── Self-attention captures long-range dependencies
├── Multi-scale feature fusion
├── Better at global context understanding
└── More compute intensive than CNNs
```

### 4. FC-EF (Fully Convolutional Early Fusion) (`models/fc_ef_*.py`)

```
Simple baseline architectures

Variants:
├── fc_ef_conc: Concatenate before/after images at input
└── fc_ef_diff: Compute difference of before/after at input

Simple but effective for smaller datasets
```

### 5. SNUNet (`models/snunet.py`)

```
Siamese Network with NestedUNet (UNet++)

Key Features:
├── Dense skip connections
├── Deep supervision (losses at multiple scales)
└── Better gradient flow
```

### 6. BiT (Binary Change Detection with Transformers) (`models/bit_cd.py`)

```
Combines CNNs with Transformers

Key Features:
├── CNN encoder for local features
├── Transformer for global context
└── Efficient attention mechanism
```

---

## 9. Loss Functions Explained

### Cross-Entropy Loss

```python
# Standard classification loss
CE = -Σ(y × log(p))

Where:
- y: ground truth (one-hot encoded)
- p: predicted probabilities

For binary classification:
BCE = -(y×log(p) + (1-y)×log(1-p))

# Penalizes confident wrong predictions heavily
```

### Dice Loss

```python
# Based on Dice coefficient (similar to F1)
Dice = 2×|A∩B| / (|A| + |B|)

DiceLoss = 1 - Dice

# Good for imbalanced segmentation
# Directly optimizes overlap between prediction and ground truth
```

### Combined BCE + Dice Loss (Used in this project)

```python
class BCEandDiceLoss(nn.Module):
    def forward(self, preds, label):
        bce_loss = CrossEntropyLoss(preds, label)
        dice_loss = DiceLoss(preds, label)
        
        return bce_loss + dice_loss

# Why combine?
# BCE: Pixel-level accuracy
# Dice: Region-level overlap
# Together: Best of both worlds
```

### Focal Loss (For extreme imbalance)

```python
FocalLoss = -α × (1-p)^γ × log(p)

Where:
- α: Class balancing weight
- γ: Focusing parameter (usually 2)
- (1-p)^γ: Down-weights easy examples

# When model is confident (p→1), loss→0
# Focuses learning on hard examples
```

---

## 10. Interview Questions & Answers

### Basic Questions

**Q: What is semantic segmentation?**
> A: Semantic segmentation is classifying every pixel in an image into predefined categories. Unlike image classification (one label per image) or object detection (bounding boxes), segmentation produces a dense prediction map where each pixel gets a class label.

**Q: Why use an encoder-decoder architecture?**
> A: The encoder extracts hierarchical features while reducing spatial dimensions (captures "what"). The decoder recovers spatial resolution (recovers "where"). Skip connections combine both, giving us detailed segmentation maps with semantic understanding.

**Q: What is transfer learning and why is it useful?**
> A: Transfer learning uses knowledge from one task to help another. Here, we use ImageNet-pretrained encoders because:
> 1. They already understand low-level features (edges, textures)
> 2. Requires less data to achieve good results
> 3. Faster convergence during training

**Q: Explain the difference between training and evaluation mode in PyTorch.**
> A: 
> - `model.train()`: Enables dropout, BatchNorm uses batch statistics
> - `model.eval()`: Disables dropout, BatchNorm uses running statistics
> 
> Forgetting to call `model.eval()` before testing is a common bug that leads to poor evaluation performance!

### Intermediate Questions

**Q: Why use IoU instead of accuracy for segmentation?**
> A: With class imbalance (e.g., 5% burnt, 95% unburnt), a model predicting everything as "unburnt" gets 95% accuracy but is useless. IoU measures overlap between prediction and ground truth, penalizing both false positives and false negatives fairly.

**Q: How does batch normalization help training?**
> A: BatchNorm normalizes layer inputs across the batch, which:
> 1. Reduces internal covariate shift
> 2. Allows higher learning rates
> 3. Acts as regularization
> 4. Makes training more stable

**Q: What is the vanishing gradient problem and how is it addressed?**
> A: In deep networks, gradients can become extremely small during backpropagation, preventing early layers from learning. Solutions:
> 1. ReLU activation (doesn't saturate for positive values)
> 2. Skip/residual connections (gradient highways)
> 3. BatchNorm (keeps activations in good range)
> 4. Proper initialization (Xavier, He)

**Q: Explain Siamese networks for change detection.**
> A: Siamese networks use the same encoder (shared weights) to process both before and after images. This ensures:
> 1. Both images are encoded in the same feature space
> 2. Fair comparison between temporal images
> 3. The model learns what changes, not image-specific biases

### Advanced Questions

**Q: How would you handle a scenario where the model has high precision but low recall?**
> A: High precision, low recall means few false positives but many missed detections. Solutions:
> 1. Lower the classification threshold
> 2. Increase weight for positive class in loss
> 3. Use focal loss to focus on hard examples
> 4. Oversample positive examples
> 5. Add more positive training examples if possible

**Q: What is mixed precision training and why use it?**
> A: Mixed precision uses FP16 instead of FP32 for most operations while keeping critical computations in FP32. Benefits:
> 1. ~2x faster training on modern GPUs
> 2. ~50% memory reduction
> 3. Allows larger batch sizes
> 4. Minimal accuracy loss when done correctly

**Q: How would you debug a model that's not learning (loss not decreasing)?**
> A: Systematic debugging approach:
> 1. Check data: visualize inputs and labels, verify normalization
> 2. Simplify: overfit on a single batch first
> 3. Learning rate: try different values (1e-3, 1e-4, 1e-5)
> 4. Gradients: check for NaN/inf values
> 5. Architecture: start with proven baseline
> 6. Loss function: verify it's computed correctly
> 7. Data loading: ensure shuffling, augmentation work

**Q: Explain the bias-variance tradeoff in the context of this project.**
> A: 
> - **High bias (underfitting)**: Model too simple, can't capture burnt area patterns. Solution: increase model capacity, more layers.
> - **High variance (overfitting)**: Model memorizes training fires, fails on new ones. Solution: regularization, dropout, data augmentation, early stopping.
> - **Goal**: Find the sweet spot where model generalizes well to unseen wildfires.

---

## 11. Common Terms Glossary

| Term | Definition |
|------|------------|
| **Backbone** | Pre-trained encoder network (ResNet, VGG, etc.) |
| **Batch Size** | Number of samples processed before updating weights |
| **Checkpoint** | Saved model state (weights, optimizer state, epoch) |
| **Epoch** | One complete pass through the training dataset |
| **Feature Map** | Output of a convolutional layer; learned representations |
| **Fine-tuning** | Training a pre-trained model on new data |
| **Ground Truth** | The correct labels/annotations |
| **Hyperparameters** | Settings chosen before training (LR, batch size, etc.) |
| **Inference** | Using trained model to make predictions |
| **Iteration** | One forward-backward pass on a batch |
| **Latent Space** | Compressed representation in bottleneck |
| **Logits** | Raw model outputs before softmax |
| **Overfitting** | Model performs well on training, poorly on validation |
| **Pooling** | Downsampling operation (max pooling, avg pooling) |
| **Receptive Field** | Region of input that affects a particular feature |
| **Tensor** | Multi-dimensional array (PyTorch's data structure) |
| **Upsampling** | Increasing spatial resolution (bilinear, transposed conv) |
| **Weight Decay** | L2 regularization on model weights |

### Tensor Shape Conventions

```
(B, C, H, W) - Batch, Channels, Height, Width

Examples:
- Input image:  (8, 9, 256, 256)  = 8 images, 9 bands, 256×256 pixels
- Feature map:  (8, 64, 128, 128) = 8 images, 64 features, 128×128 spatial
- Output:       (8, 2, 256, 256)  = 8 images, 2 classes, 256×256 pixels
```

---

## 🎯 Final Interview Tips

1. **Explain your thinking process**: Walk through the pipeline step by step
2. **Connect theory to practice**: Reference this project's implementation
3. **Know the tradeoffs**: Every decision has pros and cons
4. **Be honest**: If you don't know something, say "I'd need to look that up"
5. **Relate to real-world impact**: This project helps monitor environmental disasters

---

**Good luck with your interview! 🚀**

*Document created: February 2026*
*Project: Wildfire Impact Detection using Deep Learning*
