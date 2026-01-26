# 🔥 Wildfire Impact Detection: A Complete Guide for Aspiring Engineers

**Welcome, future engineer!** This document is your comprehensive companion to understanding not just *what* this project does, but *why* it exists, *how* it works, and what lessons you can take forward into your engineering career. I've written this as if we're sitting together, and I'm walking you through my codebase with a whiteboard nearby.

---

## 📋 Table of Contents

1. [The Big Picture: Why This Project Exists](#1-the-big-picture-why-this-project-exists)
2. [Technical Architecture: The Engine Under the Hood](#2-technical-architecture-the-engine-under-the-hood)
3. [Codebase Walkthrough: Your Map to the Code](#3-codebase-walkthrough-your-map-to-the-code)
4. [Technology Choices: The "Why" Behind Every Decision](#4-technology-choices-the-why-behind-every-decision)
5. [Bugs, Mistakes & Debugging Lessons](#5-bugs-mistakes--debugging-lessons)
6. [Engineering Mindset & Best Practices](#6-engineering-mindset--best-practices)
7. [Pitfalls & Future Improvements](#7-pitfalls--future-improvements)
8. [Final Thoughts: Becoming a Better Engineer](#8-final-thoughts-becoming-a-better-engineer)

---

## 1. The Big Picture: Why This Project Exists

### 🌍 The Real-World Problem

Every year, wildfires destroy millions of hectares of forest, displace communities, kill wildlife, and release massive amounts of carbon into the atmosphere. In 2023 alone, global wildfires burned an area larger than the state of Florida.

**Here's the challenge**: After a wildfire, emergency responders, environmental scientists, and governments need to know *exactly* what burned. They need to:
- Prioritize recovery efforts
- Assess ecological damage
- Plan reforestation
- Allocate emergency funds
- Track carbon emissions

Traditionally, humans would manually analyze satellite images or even survey areas on foot. This is:
- **Slow** (days or weeks per event)
- **Expensive** (requires trained analysts)
- **Inconsistent** (human fatigue leads to errors)
- **Dangerous** (ground surveys in burned areas)

### 💡 What This Project Does

This project uses **deep learning** to automatically detect burnt areas from satellite images. Think of it like this:

> **Analogy**: Imagine you're playing "spot the difference" between two photos—one taken before a fire and one after. A human can do this, but it's tedious. This project trains a computer to play that game at superhuman speed and scale.

More specifically, this project:

1. **Takes two satellite images** (before and after an event)
2. **Feeds them through a neural network**
3. **Outputs a map** showing exactly which pixels are burnt

The project has been extended to also detect **drought stress** in vegetation, making it a "multi-hazard" detection system.

### 🎯 Why It Matters (Beyond the Code)

This isn't just an academic exercise. The FLOGA dataset (which this project uses) contains **326 real wildfire events** from Greece between 2017-2021, with ground-truth labels created by the **Hellenic Fire Service**. Real data. Real fires. Real impact.

When you run this code, you're working with technology that could:
- Help first responders prioritize areas after a disaster
- Enable insurance companies to assess damage faster
- Help scientists track climate change effects
- Support conservation efforts worldwide

### 🔗 How Everything Fits Together (30,000 Foot View)

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLOGA Project Flow                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                    ┌───────────────────┐
│  RAW SATELLITE    │                    │   GROUND TRUTH    │
│  IMAGERY (HDF5)   │                    │   LABELS          │
│  • Sentinel-2     │                    │   (Fire Service)  │
│  • MODIS          │                    │                   │
└─────────┬─────────┘                    └─────────┬─────────┘
          │                                        │
          └───────────────┬────────────────────────┘
                          ▼
            ┌───────────────────────────┐
            │    create_dataset.py      │
            │    (Data Preprocessing)   │
            │    • Crops patches        │
            │    • Filters bad data     │
            │    • Splits train/val/test│
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │   Processed Dataset       │
            │   (.npy files + .pkl)     │
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │    run_experiment.py      │
            │    (Training/Evaluation)  │
            │    • Loads model          │
            │    • Trains on data       │
            │    • Saves best weights   │
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │   Trained Model (.pt)     │
            │   (Can predict new fires) │
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │  visualize_predictions.py │
            │  (See what model learned) │
            └───────────────────────────┘
```

---

## 2. Technical Architecture: The Engine Under the Hood

### 🏗️ System Design Philosophy

Before diving into code, let's understand *why* the system is designed this way.

**Problem**: We need to compare two images and find what changed.

**Naive Solution**: Just subtract the images?

**Why That Fails**: 
- Lighting changes between dates (clouds, sun angle)
- Seasonal vegetation changes (autumn vs summer)
- Sensor differences and noise
- Not all dark pixels are burnt—shadows exist!

**Better Solution**: Use a neural network that *learns* what "burnt" looks like by studying thousands of examples.

### 🧠 The Core Concept: Change Detection

This project does **semantic change detection**—a fancy term for "look at two images and label what changed, pixel by pixel."

> **Analogy**: Think of this like a medical imaging system. A radiologist compares two brain scans taken months apart to see if a tumor grew. Our system compares two satellite images to see if vegetation burned.

#### The Siamese Architecture Pattern

Most models in this project use a **Siamese (twin) architecture**:

```
┌─────────────────┐     ┌─────────────────┐
│  Before Image   │     │  After Image    │
│  (Pre-fire)     │     │  (Post-fire)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
  ┌──────────────┐        ┌──────────────┐
  │   Encoder    │        │   Encoder    │ ← Same weights (twins!)
  │  (ResNet)    │        │  (ResNet)    │
  └──────┬───────┘        └──────┬───────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────┐
              │   Compare   │ ← Concatenate or Subtract
              │  Features   │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │   Decoder   │ ← Reconstruct spatial details
              │  (U-Net)    │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  Prediction │
              │  (Per-pixel)│
              │  0=No fire  │
              │  1=Burnt    │
              └─────────────┘
```

**Why Siamese?** By using the *same* encoder for both images, the network learns features that are directly comparable. It's like using the same ruler to measure two objects—you can trust the comparison.

### 📡 Data Flow: From Satellite to Prediction

Let's trace how data flows through the system:

#### Stage 1: Raw Data (HDF5 Files)

Satellite data arrives as compressed HDF5 files. Each file contains:
- **Sentinel-2 imagery**: 10 spectral bands at various resolutions (10m, 20m, 60m)
- **MODIS imagery**: 7 bands at 500m resolution
- **Masks**: Cloud masks, water masks, land cover
- **Labels**: Ground truth burnt area maps

> **Why HDF5?** It's a scientific data format that handles large arrays efficiently. It's like a ZIP file for numerical data, but you can read parts without extracting everything.

#### Stage 2: Preprocessing (`create_dataset.py`)

Raw imagery needs cleaning:

```python
# What preprocessing does:
1. Crop large images into 256×256 patches (manageable for GPU memory)
2. Remove patches that are mostly sea/water
3. Remove patches that are mostly cloudy
4. Split into train/validation/test sets
5. Save as .npy files (fast to load)
```

**Why patches?** Full satellite scenes are enormous (10,000+ pixels). Neural networks work better with smaller, consistent-sized inputs. It's like reading a book one page at a time rather than the whole thing at once.

#### Stage 3: Training (`run_experiment.py`)

The training loop:

```python
for epoch in range(n_epochs):
    for batch in train_loader:
        # 1. Get before/after images and labels
        before, after, label = batch
        
        # 2. Forward pass: model makes prediction
        prediction = model(before, after)
        
        # 3. Compute loss: how wrong was the prediction?
        loss = criterion(prediction, label)
        
        # 4. Backward pass: compute gradients
        loss.backward()
        
        # 5. Update weights: learn from mistakes
        optimizer.step()
```

> **Analogy**: Training is like learning to cook. Each batch is a recipe attempt. The loss function tells you how bad it tastes. Gradient descent is figuring out "add less salt next time."

#### Stage 4: Evaluation

After training, we evaluate on held-out test data:

```python
Metrics computed:
├── F1-Score: Balance of precision and recall
├── IoU (Intersection over Union): How much prediction overlaps truth
├── Precision: Of pixels we called "burnt", how many actually were?
├── Recall: Of actually burnt pixels, how many did we find?
└── Confusion Matrix: Full breakdown of predictions vs reality
```

### 🎛️ The Configuration System

One of the smartest design decisions in this project is the **configuration-driven architecture**. Instead of hardcoding values, everything is in JSON files:

```json
{
    "method": "bam_cd",           // Which model to use
    "event_type": "wildfire",     // Wildfire or drought?
    "train": {
        "n_epochs": 150,          // How long to train
        "batch_size": 8           // How many patches per gradient update
    },
    "datasets": {
        "data_source": "sen2",    // Which satellite?
        "scale_input": "clamp_scale_10000"  // How to normalize
    }
}
```

**Why This Matters**:
- **Reproducibility**: Share a config, reproduce an experiment
- **Experimentation**: Try 10 models by changing one line
- **Debugging**: Config files serve as documentation
- **Collaboration**: Others can understand your setup instantly

---

## 3. Codebase Walkthrough: Your Map to the Code

Let's explore the codebase like we're walking through a new city. I'll point out the landmarks and explain why each neighborhood exists.

### 📁 Folder Structure Overview

```
Wildfire_Impact/
│
├── 📜 Main Scripts (the "action" happens here)
│   ├── run_experiment.py      ← The main entry point
│   ├── create_dataset.py      ← Prepares raw data
│   ├── visualize_predictions.py ← See what the model learned
│   └── create_synthetic_dataset.py ← Generate test data
│
├── 🧠 models/                  ← Neural network architectures
│   ├── bam_cd/                 ← The star: BAM-CD model
│   │   ├── model.py            ← Main model class
│   │   ├── encoder.py          ← Feature extraction
│   │   └── decoder.py          ← Spatial reconstruction
│   ├── unet.py                 ← Classic U-Net
│   ├── changeformer.py         ← Transformer-based
│   └── snunet.py               ← SNUNet with attention
│
├── 📊 losses/                  ← How we measure "wrongness"
│   ├── dice.py                 ← Dice loss (good for imbalanced data)
│   └── bce_and_dice.py         ← Combined loss
│
├── ⚙️ configs/                 ← Configuration files
│   ├── config.json             ← Main config
│   └── method/                 ← Model-specific configs
│       ├── bam_cd.json
│       └── unet.json
│
├── 📁 data/                    ← Where datasets live
│   └── processed/              ← Ready-to-use data
│       └── sen2_20_mod_500/    ← Sentinel-2 at 20m + MODIS at 500m
│
├── 📁 results/                 ← Training outputs
│   └── <model>/<timestamp>/    ← Each run gets a folder
│       └── checkpoints/        ← Saved model weights
│
├── 🔧 Utility Files
│   ├── utils.py                ← Helper functions
│   ├── dataset_utils.py        ← Data loading logic
│   └── cd_experiments_utils.py ← Training/eval loops
│
└── 📄 Documentation
    ├── README.md               ← Quick start
    ├── main.md                 ← This file!
    └── EXPLANATION.md          ← Technical deep dive
```

### 🔍 Key Files Explained (Like Reading a Story)

#### The Protagonist: `run_experiment.py`

This is where everything comes together. Think of it as the conductor of an orchestra:

```python
# Simplified structure of run_experiment.py

# 1. Parse command line arguments
args = parser.parse_args()
configs = load_config(args.config)

# 2. Set up the stage
device = 'cuda:0'  # Use GPU
train_dataset = Dataset('train', configs)
val_dataset = Dataset('val', configs)

# 3. Initialize the players
model = init_model(configs)
optimizer = init_optimizer(model, configs)
criterion = create_loss(configs)

# 4. The performance (training loop)
if configs['mode'] == 'train':
    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion)
        validate(model, val_loader)
        save_if_best(model)

# 5. The review (evaluation)
else:
    evaluate(model, test_loader)
```

**Key Insight**: Notice how modular this is. The model, optimizer, and loss are all interchangeable. Want to try a different model? Just change the config. This is called **dependency injection** and it's a core software engineering principle.

#### The Data Chef: `dataset_utils.py`

This file defines how data is loaded and preprocessed:

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, configs):
        # Load the pickle file listing all patches
        self.patches = pickle.load(open(split_file, 'rb'))
        
        # Filter by event type (wildfire or drought)
        if self.event_type != 'all':
            self.patches = filter_by_type(self.patches)
    
    def __getitem__(self, idx):
        # Load one patch
        before = np.load(self.patches[idx]['S2_before'])
        after = np.load(self.patches[idx]['S2_after'])
        label = np.load(self.patches[idx]['label'])
        
        # Apply transformations (scaling, augmentation)
        sample = self.scale_img({'before': before, 'after': after})
        
        return sample
```

**Design Pattern**: This uses PyTorch's `Dataset` pattern. The `__getitem__` method defines how to fetch one sample. The `DataLoader` class then handles batching, shuffling, and parallel loading automatically.

> **Analogy**: The `Dataset` class is like a recipe book—it knows how to make each dish. The `DataLoader` is the kitchen staff that prepares multiple dishes simultaneously.

#### The Artist: `visualize_predictions.py`

Seeing is believing. This script generates visual outputs:

```python
# For each test image:
1. Load before/after images
2. Run model prediction
3. Create visualization:
   ┌────────────┬────────────┬────────────┬────────────┐
   │  Before    │   After    │   Truth    │ Prediction │
   │  (RGB)     │   (RGB)    │  (Mask)    │   (Mask)   │
   └────────────┴────────────┴────────────┴────────────┘
4. Save to results folder
5. Organize by prediction quality (TP, FP, FN)
```

**Why This Matters**: Numbers (F1=0.87) are abstract. Seeing the model highlight the exact burnt pixels makes the results tangible and helps identify failure modes.

#### The Brains: `models/bam_cd/model.py`

The BAM-CD (Burnt Area Mapping with Change Detection) model is the project's flagship architecture:

```python
class BAM_CD(torch.nn.Module):
    def __init__(self, ...):
        # Siamese encoder (same weights for both images)
        self.encoder = ResNet34(pretrained=True)
        
        # Feature fusion (how to compare the images)
        self.fusion_mode = 'conc'  # concatenate features
        
        # U-Net decoder (reconstruct spatial details)
        self.decoder = UnetDecoder(...)
        
        # Output head (predict per-pixel classes)
        self.segmentation_head = SegmentationHead(...)
    
    def forward(self, x_pre, x_post):
        # Extract features from both images
        features1 = self.encoder(x_pre)
        features2 = self.encoder(x_post)
        
        # Combine features
        if self.fusion_mode == 'conc':
            combined = concatenate(features1, features2)
        else:
            combined = features1 - features2
        
        # Decode to full resolution
        decoded = self.decoder(combined)
        
        # Final prediction
        return self.segmentation_head(decoded)
```

**Key Design Choices**:
1. **Pretrained encoder**: Starts with ImageNet knowledge (transfer learning)
2. **Siamese architecture**: Ensures fair comparison between images
3. **SCSE attention**: Focuses on relevant features (spatial and channel-wise)
4. **Multi-scale features**: Captures both fine details and context

### 🔌 How Components Interact

Here's the call graph when you run training:

```
run_experiment.py
├── loads config.json
├── validates configs (utils.validate_configs)
├── creates Dataset objects (dataset_utils.Dataset)
│   └── loads .pkl files
│   └── sets up data augmentation
├── creates DataLoaders
├── initializes model (utils.init_model)
│   └── loads model architecture (models/bam_cd/model.py)
│   └── loads pretrained weights if specified
├── initializes optimizer (utils.init_optimizer)
├── initializes loss (utils.create_loss)
│   └── may use losses/bce_and_dice.py
├── calls train_change_detection() (cd_experiments_utils.py)
│   └── for each epoch:
│       └── for each batch:
│           └── forward pass through model
│           └── compute loss
│           └── backward pass
│           └── optimizer step
│       └── validate and save best model
└── logs metrics to console/wandb
```

---

## 4. Technology Choices: The "Why" Behind Every Decision

Great engineers don't just know *what* tools to use—they understand *why* certain tools are chosen and what tradeoffs they bring.

### 🐍 Python: The Lingua Franca of ML

**Why Python?**
- Rich ecosystem: PyTorch, NumPy, scikit-learn
- Readable syntax: Code reads like pseudocode
- Community: Massive support, endless tutorials
- Interoperability: Easy to integrate with C/C++ for speed

**Tradeoffs**:
- Slower than compiled languages (but GPU does the heavy lifting)
- Global Interpreter Lock limits multithreading (use multiprocessing instead)
- Dynamic typing can hide bugs (use type hints!)

### 🔥 PyTorch: The Deep Learning Framework

**Why PyTorch over TensorFlow?**

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| Debugging | Easy (Python-native) | Harder (graph execution) |
| Research | Preferred by academics | Preferred in production |
| Dynamic graphs | Yes (define-by-run) | Added later (eager mode) |
| Learning curve | Gentler | Steeper |

**This project's perspective**: PyTorch's flexibility is perfect for research where we experiment with many architectures. The debugging experience is invaluable when things go wrong.

### 🛰️ Sentinel-2 & MODIS: The Eyes in the Sky

**Sentinel-2**:
- Free and open (European Space Agency)
- 10m resolution for RGB bands
- 13 spectral bands (can "see" beyond visible light)
- 5-day revisit time (frequent updates)

**MODIS**:
- Lower resolution (500m) but broader coverage
- Longer historical archive (since 2000)
- Useful for regional-scale analysis

**Why both?** Different tools for different scales. Sentinel-2 for detailed mapping, MODIS for rapid assessment over large areas.

### 📊 NumPy Arrays: The Data Structure Choice

**Why .npy files instead of images (PNG/JPEG)?**

```python
# Satellite data has 9 channels (not 3 like RGB)
shape = (9, 256, 256)  # Channels × Height × Width

# Each pixel stores reflectance values (not colors)
dtype = np.int16  # Precise numerical values, not 8-bit colors

# Loading is fast
data = np.load('patch.npy')  # Direct memory mapping possible
```

**Tradeoffs**:
- Larger file size than compressed formats
- Not human-viewable (need code to visualize)
- But: Preserves scientific precision perfectly

### 🎯 Cross-Entropy + Dice Loss: The Training Signal

**Why not just Cross-Entropy?**

The dataset is **imbalanced**: Most pixels are NOT burnt. With vanilla cross-entropy:

```
Dataset: 95% non-burnt, 5% burnt
Dumb model: "Everything is non-burnt!" 
Accuracy: 95%! (but completely useless)
```

**Dice Loss** specifically measures overlap between prediction and ground truth, penalizing the model for missing burnt areas regardless of how many non-burnt pixels it gets right.

**The combination**: Cross-entropy provides stable gradients early in training; Dice loss ensures we don't ignore the minority class.

### 📁 Configuration Files (JSON): The Experiment Tracker

**Why JSON configs instead of command-line arguments?**

```bash
# Hard to remember, easy to make mistakes:
python train.py --model bam_cd --epochs 150 --batch_size 8 --lr 0.001 --...

# Clear, versionable, shareable:
python run_experiment.py --config configs/experiment_42.json
```

**Best Practice**: Commit config files to git. Each experiment becomes reproducible forever.

### 🏃 Mixed Precision Training: Speed Optimization

```python
configs['train']['mixed_precision'] = True
```

**What it does**: Uses 16-bit floats (half precision) for most operations, 32-bit only where needed.

**Benefits**:
- 2× faster training
- Half the GPU memory usage
- Enables larger batch sizes

**Tradeoffs**:
- Slight numerical instability (mitigated by loss scaling)
- Not all operations support it (rare edge cases)

---

## 5. Bugs, Mistakes & Debugging Lessons

**This section is crucial.** Bugs are not failures—they're learning opportunities. Every bug you encounter and solve makes you a better engineer.

### 🐛 Bug #1: The Vanishing Checkpoint

**Symptom**: Training completes but no `best_segmentation.pt` file is saved.

**Investigation**:
```python
# Original code
if val_f1 > best_val:
    save_checkpoint(model)
    best_val = val_f1
```

**Root Cause**: `val_f1` started at 0.0, but the validation set had a bug where all predictions were garbage (random). The model never beat 0.0 because the validation never ran correctly.

**The Fix**: 
```python
# Add defensive logging
print(f"Val F1: {val_f1}, Best: {best_val}")
if val_f1 > best_val:
    print(f"New best! Saving checkpoint...")
    save_checkpoint(model)
```

**Lesson**: Always log intermediate values. Silent failures are the worst kind. If something should happen (like saving a checkpoint), log when it does AND when it doesn't.

---

### 🐛 Bug #2: The `positive_flag` KeyError

**Symptom**: `KeyError: 'positive_flag'` when loading older datasets.

**Investigation**:
```python
# Newer datasets have this field
patch_info = {'positive_flag': True, 'path': '...'}

# Older datasets don't
patch_info = {'path': '...'}  # No positive_flag!
```

**Root Cause**: The code was updated to use a new field, but old dataset files didn't have it.

**The Fix** (defensive programming):
```python
# Before (brittle)
is_positive = patch_info['positive_flag']

# After (defensive)
is_positive = patch_info.get('positive_flag', False)  # Default to False
```

**Lesson**: When loading external data, NEVER assume fields exist. Use `.get()` with defaults. This is called **defensive programming**—assume the world is trying to break your code.

---

### 🐛 Bug #3: The Mysterious NaN Loss

**Symptom**: Loss becomes `NaN` after a few epochs, training collapses.

**Investigation**:
```python
# Debug print
print(f"Before img stats: min={before.min()}, max={before.max()}")
# Output: Before img stats: min=-28672, max=65535
```

**Root Cause**: Some satellite pixels had special "no data" values (-28672 for MODIS). When normalized, these became huge or infinite values, poisoning the network.

**The Fix**:
```python
# Handle special values before normalization
NODATA_VALUES = {'SEN2': 0, 'MOD': -28672}

def preprocess(img, source):
    nodata = NODATA_VALUES[source]
    img[img == nodata] = 0  # Replace with valid value
    return img
```

**Lesson**: Always inspect your data! `print()` statements are your friends during debugging. Real-world data is MESSY.

---

### 🐛 Bug #4: Evaluation Mode Mishap

**Symptom**: Model performs great during training, terribly during evaluation.

**Investigation**: Training accuracy 90%, test accuracy 45%?!

**Root Cause**: Forgot to call `model.eval()` before evaluation.

```python
# PyTorch models behave differently in train vs eval mode
# Train mode: Dropout active, BatchNorm uses batch statistics
# Eval mode: Dropout disabled, BatchNorm uses running statistics

# The bug
predictions = model(test_images)  # Still in train mode!

# The fix
model.eval()  # CRITICAL!
with torch.no_grad():  # Also saves memory
    predictions = model(test_images)
```

**Lesson**: Always explicitly set `model.train()` and `model.eval()`. Make it part of your mental checklist. This bug has bitten every deep learning practitioner at least once.

---

### 🐛 Bug #5: The Off-By-One Class Label

**Symptom**: Metrics look weird. Precision for "burnt" class is always 0.

**Investigation**:
```python
# Ground truth labels
unique_labels = np.unique(labels)
print(unique_labels)  # [0, 1, 2]

# Class 2 = "Other burnt events" (should be ignored!)
# But the loss function was treating it as a valid class
```

**Root Cause**: The loss function had `ignore_index=2` but the metric calculation didn't.

**The Fix**: Ensure ALL components (loss, metrics, visualization) use the same `ignore_index`:
```python
cm = ConfusionMatrix(num_classes=2, ignore_index=2)  # Ignore class 2
iou = JaccardIndex(num_classes=2, ignore_index=2)    # Same here!
```

**Lesson**: When multiple components need the same configuration, centralize it. Don't repeat yourself (DRY principle).

---

### 🐛 Bug #6: The GPU Memory Leak

**Symptom**: Training starts fine but crashes with "CUDA out of memory" after a few epochs.

**Investigation**: GPU memory usage slowly climbs even though batch size is constant.

**Root Cause**: Accumulating tensors in a list for logging:
```python
# The bug
all_losses = []
for batch in train_loader:
    loss = compute_loss(...)
    all_losses.append(loss)  # Keeps entire computation graph!

# The fix
all_losses = []
for batch in train_loader:
    loss = compute_loss(...)
    all_losses.append(loss.item())  # .item() extracts the number, frees graph
```

**Lesson**: In PyTorch, tensors carry their computation history (for backprop). When logging, always use `.item()` to extract raw Python numbers. This is a classic footgun.

---

## 6. Engineering Mindset & Best Practices

Beyond code, let's talk about how great engineers *think*.

### 🧩 How to Break Down Problems

**The Wrong Way**: "I need to build a wildfire detection system." (Too vague!)

**The Right Way** (Decomposition):
```
1. How do I load satellite data?
   ├── What format is it in? (HDF5)
   ├── What library reads HDF5? (h5py)
   └── What's the array shape? (C, H, W)

2. How do I prepare data for training?
   ├── What size patches? (256×256)
   ├── How to handle edges? (Padding)
   └── How to split data? (60/20/20)

3. What model architecture?
   ├── What's the input? (Two images)
   ├── What's the output? (Per-pixel labels)
   └── What existing architectures work? (U-Net, etc.)

... and so on
```

> **Rule of Thumb**: If you can't explain what a function should do in one sentence, it's doing too much. Break it down further.

### 📝 Code Readability Principles

**Variable Names Matter**:
```python
# Bad
x1 = load('before.npy')
x2 = load('after.npy')
y = model(x1, x2)

# Good
before_image = load('before.npy')
after_image = load('after.npy')
prediction = model(before_image, after_image)
```

**Comments Explain "Why", Not "What"**:
```python
# Bad comment (describes what's obvious)
x = x + 1  # Add 1 to x

# Good comment (explains why)
x = x + 1  # Account for 0-based indexing in Python
```

**Magic Numbers Are Evil**:
```python
# Bad
if pixel_value > 10000:
    pixel_value = 10000

# Good
SENTINEL2_MAX_REFLECTANCE = 10000
if pixel_value > SENTINEL2_MAX_REFLECTANCE:
    pixel_value = SENTINEL2_MAX_REFLECTANCE
```

### 🔍 Systematic Debugging

When something doesn't work, resist the urge to randomly change things. Instead:

1. **Reproduce**: Can you make it fail consistently?
2. **Isolate**: What's the smallest code that fails?
3. **Hypothesize**: What could cause this?
4. **Test**: Add print statements or breakpoints
5. **Fix**: Make ONE change at a time
6. **Verify**: Does the original problem go away?
7. **Regression Test**: Did you break anything else?

**Debug Print Template**:
```python
def suspicious_function(data):
    print(f"[DEBUG] Input shape: {data.shape}, dtype: {data.dtype}")
    print(f"[DEBUG] Input range: [{data.min()}, {data.max()}]")
    
    result = complex_operation(data)
    
    print(f"[DEBUG] Output shape: {result.shape}")
    print(f"[DEBUG] Output contains NaN: {np.isnan(result).any()}")
    
    return result
```

### 🏗️ Design Patterns You'll See

**Factory Pattern** (in `utils.py`):
```python
def init_model(configs, ...):
    if configs['method'] == 'unet':
        return Unet(...)
    elif configs['method'] == 'bam_cd':
        return BAM_CD(...)
    # ... etc
```
*One function that creates different objects based on input.*

**Strategy Pattern** (loss functions):
```python
# Different strategies for computing loss
losses = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'dice': DiceLoss(),
    'dice+ce': BCEandDiceLoss()
}
criterion = losses[configs['train']['loss_function']]
```
*Swappable algorithms with the same interface.*

**Observer Pattern** (Weights & Biases logging):
```python
# Training loop doesn't care HOW metrics are logged
# It just notifies observers
if configs['wandb']['activate']:
    wandb.log({'loss': loss, 'f1': f1})
```
*Decouple the "what happened" from "who needs to know".*

### 🧪 Testing Philosophy

**Why This Project Has Synthetic Data**:

Downloading the full FLOGA dataset takes hours and requires 50GB+. But you should be able to test the code in 5 minutes.

```python
# create_synthetic_dataset.py generates fake but structurally correct data
# This lets you:
# 1. Test the entire pipeline without real data
# 2. Catch bugs early (before expensive training)
# 3. Run CI/CD pipelines quickly
```

**Principle**: Make it easy to test. If testing is painful, it won't happen.

---

## 7. Pitfalls & Future Improvements

### ⚠️ Common Mistakes to Avoid

**Mistake #1: Not Setting Random Seeds**
```python
# Without seeds, results vary between runs
# Makes debugging nearly impossible

# Always set seeds early
np.random.seed(999)
torch.manual_seed(999)
random.seed(999)
```

**Mistake #2: Training on Test Data**
```python
# This is research fraud, but also easy to do accidentally
# NEVER look at test data during development
# Test set is SACRED - only touch it for final evaluation
```

**Mistake #3: Ignoring Class Imbalance**
```python
# Dataset: 95% non-burnt, 5% burnt
# Model learns to predict "not burnt" everywhere
# Solution: Use weighted loss, oversample positive patches
```

**Mistake #4: Not Checking Data Augmentation**
```python
# Augmentation can create invalid samples
# Example: Horizontal flip is fine for wildfires
# But: Rotation might make fire spread patterns unrealistic
# Always VISUALIZE augmented samples!
```

**Mistake #5: Overfitting to Validation Set**
```python
# If you tune hyperparameters on val set 100 times,
# you're essentially training on the val set
# Solution: Use a separate held-out set for final model selection
```

### 🚀 Future Improvements

**1. Multi-GPU Training**
Currently limited to single GPU. For larger datasets:
```python
# Could add:
model = nn.DataParallel(model)  # Simple but inefficient
# Or better:
model = nn.DistributedDataParallel(model)  # Scales well
```

**2. Better Cloud Handling**
Clouds obscure satellite imagery. Current approach: skip cloudy patches.
Better approach: Cloud removal algorithms or multi-date compositing.

**3. Real-time Inference Pipeline**
Current workflow: Research (batch processing).
Production would need: Streaming pipeline for new satellite data.

**4. Uncertainty Quantification**
Current model: "This pixel is burnt" (binary).
Better: "This pixel is burnt with 95% confidence" (probabilistic).

**5. Active Learning**
Let the model ask for labels on confusing cases, reducing labeling effort.

**6. Explainability**
Add attention visualization to understand *why* the model made a prediction.

### 📈 Scalability Considerations

**Memory Bottlenecks**:
- GPU memory limits batch size
- Solution: Gradient accumulation, mixed precision

**Storage Bottlenecks**:
- Full dataset is 50GB+
- Solution: Stream data from cloud storage (S3, GCS)

**Compute Bottlenecks**:
- Training takes hours/days
- Solution: Distributed training, TPU clusters

---

## 8. Final Thoughts: Becoming a Better Engineer

### 🎓 Key Takeaways

1. **Understand the problem before coding**: This project exists because manual fire mapping is slow and dangerous. The code serves a purpose.

2. **Design for change**: The config-driven architecture means experiments are easy. Flexibility is a feature.

3. **Bugs are teachers**: Every bug you fix teaches you something. Document them, learn from them.

4. **Read code actively**: Don't just skim. Ask "why is this here?" at every line.

5. **Build iteratively**: Start simple (synthetic data, small model), then scale up.

### 📚 What to Study Next

If this project excites you:

1. **Deep Learning Basics**: Andrew Ng's Coursera course
2. **Computer Vision**: CS231n (Stanford, free online)
3. **Remote Sensing**: "Remote Sensing Digital Image Analysis" by Richards
4. **Software Engineering**: "Clean Code" by Robert Martin

### 💭 A Final Thought

This codebase isn't perfect. No codebase is. But it solves a real problem, and it's structured well enough that you can understand it, modify it, and build on it.

That's what good engineering is about: **solving real problems with tools that others can use and improve**.

Now go break something, fix it, and learn from the process. That's how engineers are made.

---

*Good luck, future engineer. The world needs people who can see satellites above and burnt forests below, and build bridges between them with code.*

---

## 📎 Quick Reference Card

### Essential Commands
```bash
# Generate test data
python create_synthetic_dataset.py --event_type wildfire

# Run training
python run_experiment.py --config configs/config.json

# Evaluate model
python run_experiment.py --config configs/config_eval.json

# Visualize predictions
python visualize_predictions.py --config configs/config.json --mode test
```

### Key Files to Read First
1. `run_experiment.py` - The orchestrator
2. `models/bam_cd/model.py` - The neural network
3. `dataset_utils.py` - Data loading
4. `configs/config.json` - Configuration structure

### Debugging Checklist
- [ ] Is `model.eval()` called for evaluation?
- [ ] Are random seeds set?
- [ ] Is the data normalized correctly?
- [ ] Are there NaN values anywhere?
- [ ] Is GPU memory leaking?
- [ ] Is the loss decreasing?
- [ ] Do the shapes match at each layer?

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Author**: A fellow engineer who's been where you are
