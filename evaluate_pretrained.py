"""
Evaluate pre-trained model on synthetic dataset
"""
import pyjson5
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import init_model, font_colors
from dataset_utils import Dataset
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall

# Load configs
config_path = 'configs/config_eval_synthetic.json'
configs = pyjson5.load(open(config_path, 'r'))
method_config_path = Path('configs') / 'method' / f"{configs['method']}.json"
model_configs = pyjson5.load(open(method_config_path, 'r'))

print(f"{font_colors.BOLD}=" * 60)
print(f"Evaluating Pre-trained BAM-CD Model")
print(f"=" * 60 + f"{font_colors.ENDC}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load model
model_path = configs['paths']['load_state']
print(f"\nLoading model from: {model_path}")

checkpoint = torch.load(model_path, map_location=device)
print(f"Checkpoint epoch: {checkpoint['epoch']}")

# Initialize model
num_channels = 9  # Sentinel-2 20m resolution
model = init_model(configs, model_configs, checkpoint, num_channels, device)
model = model[0]  # Extract model from tuple
model.eval()
print(f"✅ Model loaded successfully\n")

# Load test dataset
print("Loading test dataset...")

test_dataset = Dataset(
    mode='test',
    configs=configs,
    clc=False,
    clouds=False,
    sea=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {len(test_dataset)}\n")

# Initialize metrics
iou_metric = BinaryJaccardIndex().to(device)
f1_metric = BinaryF1Score().to(device)
precision_metric = BinaryPrecision().to(device)
recall_metric = BinaryRecall().to(device)

# Evaluate
print(f"{font_colors.BOLD}Running evaluation...{font_colors.ENDC}\n")

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Get inputs (find S2_before and S2_after keys)
        before_key = [k for k in batch.keys() if 'before' in k and 'S2' in k][0]
        after_key = [k for k in batch.keys() if 'after' in k and 'S2' in k][0]
        label_key = 'label'
        
        before = batch[before_key].to(device)
        after = batch[after_key].to(device)
        label = batch[label_key].to(device).long()
        
        # Forward pass
        output = model(before, after)
        
        # Get predictions (class 1 = burnt)
        pred = torch.argmax(output, dim=1)
        
        # Store for metrics
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())
        
        # Update metrics
        iou_metric.update(pred, label)
        f1_metric.update(pred, label)
        precision_metric.update(pred, label)
        recall_metric.update(pred, label)

# Compute final metrics
iou = iou_metric.compute().item()
f1 = f1_metric.compute().item()
precision = precision_metric.compute().item()
recall = recall_metric.compute().item()

# Print results
print(f"\n{font_colors.BOLD}{'=' * 60}")
print(f"EVALUATION RESULTS")
print(f"{'=' * 60}{font_colors.ENDC}")
print(f"\n{font_colors.GREEN}IoU (Jaccard Index):{font_colors.ENDC} {iou:.4f}")
print(f"{font_colors.GREEN}F1 Score:{font_colors.ENDC}           {f1:.4f}")
print(f"{font_colors.GREEN}Precision:{font_colors.ENDC}          {precision:.4f}")
print(f"{font_colors.GREEN}Recall:{font_colors.ENDC}             {recall:.4f}")
print(f"\n{font_colors.BOLD}{'=' * 60}{font_colors.ENDC}\n")

print(f"{font_colors.CYAN}Note: These are synthetic test samples.")
print(f"For real performance, evaluate on the full FLOGA dataset.{font_colors.ENDC}\n")
