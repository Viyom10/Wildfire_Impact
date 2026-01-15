"""
Create a minimal synthetic dataset for testing FLOGA pipeline
This generates fake Sentinel-2 images and labels to test the evaluation pipeline
"""
import numpy as np
import pickle
from pathlib import Path
import torch

print("=" * 60)
print("Creating Minimal Test Dataset")
print("=" * 60)

# Configuration
OUTPUT_PATH = Path("data/processed")
DATASET_TYPE = "sen2_20_mod_500"
PATCH_SIZE = 256
NUM_BANDS = 9  # Sentinel-2 at 20m resolution

# Create directory structure
dataset_path = OUTPUT_PATH / DATASET_TYPE
dataset_path.mkdir(parents=True, exist_ok=True)

# Create a few synthetic events
NUM_TRAIN = 10
NUM_VAL = 3
NUM_TEST = 3

def create_synthetic_patch(patch_id, is_positive=True):
    """Create a synthetic Sentinel-2 patch with label"""
    # Create before/after images (9 bands, 256x256)
    before = np.random.randint(0, 10000, (NUM_BANDS, PATCH_SIZE, PATCH_SIZE), dtype=np.int16)
    after = before.copy()
    
    # Create label
    label = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    
    if is_positive:
        # Add some "burnt" areas in after image and label
        # Create a circular burnt area in the center
        center_x, center_y = PATCH_SIZE // 2, PATCH_SIZE // 2
        radius = PATCH_SIZE // 4
        
        y, x = np.ogrid[:PATCH_SIZE, :PATCH_SIZE]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Make after image darker in burnt area (simulating burnt vegetation)
        after[:, mask] = (after[:, mask] * 0.5).astype(np.int16)
        
        # Set label to 1 (burnt)
        label[mask] = 1
    
    return before, after, label

print("\nCreating synthetic patches...")

# Generate train/val/test splits
splits = {
    'train': NUM_TRAIN,
    'val': NUM_VAL,
    'test': NUM_TEST
}

for split_name, num_samples in splits.items():
    print(f"\nCreating {split_name} split ({num_samples} samples)...")
    
    split_data = {}
    
    for i in range(num_samples):
        # Alternate between positive and negative samples
        is_positive = (i % 2 == 0)
        
        event_id = f"synthetic_event_{split_name}_{i:03d}"
        patch_id = f"{event_id}_patch_000"
        
        # Create synthetic data
        before, after, label = create_synthetic_patch(patch_id, is_positive)
        
        # Save as numpy arrays
        year_path = OUTPUT_PATH / "2024"  # Use a synthetic year
        year_path.mkdir(exist_ok=True)
        
        before_path = year_path / f"{patch_id}.S2_before.npy"
        after_path = year_path / f"{patch_id}.S2_after.npy"
        label_path = year_path / f"{patch_id}.label.npy"
        
        np.save(before_path, before)
        np.save(after_path, after)
        np.save(label_path, label)
        
        # Add to split dictionary
        split_data[patch_id] = {
            'S2_before_image': before_path,
            'S2_after_image': after_path,
            'label': label_path,
            'positive_flag': is_positive,
            'event_id': event_id,
            'sample_key': patch_id
        }
        
        print(f"  ✓ Created {patch_id} ({'positive' if is_positive else 'negative'})")
    
    # Save pickle file
    pkl_path = dataset_path / f"synthetic_{split_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"  ✓ Saved {pkl_path}")

print("\n" + "=" * 60)
print("✅ Synthetic dataset created successfully!")
print("=" * 60)
print(f"\nDataset location: {OUTPUT_PATH}")
print(f"\nSplit files:")
print(f"  - {dataset_path / 'synthetic_train.pkl'}")
print(f"  - {dataset_path / 'synthetic_val.pkl'}")
print(f"  - {dataset_path / 'synthetic_test.pkl'}")
print(f"\nTotal patches: {NUM_TRAIN + NUM_VAL + NUM_TEST}")
print(f"Dataset size: ~{(NUM_TRAIN + NUM_VAL + NUM_TEST) * PATCH_SIZE * PATCH_SIZE * NUM_BANDS * 2 * 2 / (1024**2):.1f} MB")
print("\nYou can now run evaluation with:")
print("  python run_experiment.py --config configs/config_eval_synthetic.json")
print("=" * 60)
