"""
Wildfire Impact Detection from Satellite Images

This script allows you to upload satellite images directly and get burn area predictions.
It supports multiple image formats and handles all necessary preprocessing.

Usage Examples:
--------------
1. Using two separate images (before and after):
   python predict_from_images.py --before path/to/before_image.tif --after path/to/after_image.tif

2. Using RGB images (will be replicated to 9 channels for demo):
   python predict_from_images.py --before before.png --after after.png --rgb

3. Using GeoTIFF with multiple bands:
   python predict_from_images.py --before before.tif --after after.tif --bands 9

4. Save output visualization:
   python predict_from_images.py --before before.tif --after after.tif --output results/prediction.png

5. Run demo with sample data:
   python predict_from_images.py --demo

Supported Input Formats:
- GeoTIFF (.tif, .tiff) - preferred for satellite imagery
- NumPy arrays (.npy) - direct array input
- Standard images (.png, .jpg, .jpeg) - will be converted

The model expects 9-channel Sentinel-2 imagery at 20m resolution:
Bands: B02, B03, B04, B05, B06, B07, B11, B12, B8A
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import pyjson5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Import model utilities
from utils import init_model, font_colors


def load_image(image_path, expected_channels=9, target_size=(256, 256)):
    """
    Load an image from various formats and convert to model-ready format.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    expected_channels : int
        Number of channels expected by the model (default 9 for Sentinel-2 20m)
    target_size : tuple
        Target spatial dimensions (height, width)
    
    Returns:
    --------
    numpy.ndarray : Image array of shape (channels, height, width)
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    suffix = image_path.suffix.lower()
    
    # Load based on file type
    if suffix == '.npy':
        # NumPy array
        img = np.load(image_path).astype(np.float32)
        
    elif suffix in ['.tif', '.tiff']:
        # GeoTIFF - try rasterio first, then PIL
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                img = src.read().astype(np.float32)  # Shape: (bands, height, width)
        except ImportError:
            # Fall back to PIL/tifffile
            try:
                import tifffile
                img = tifffile.imread(image_path).astype(np.float32)
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]  # Add channel dim
                elif img.ndim == 3 and img.shape[2] <= 12:
                    # Assume (H, W, C) format, convert to (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
            except ImportError:
                # Use PIL as last resort
                pil_img = Image.open(image_path)
                img = np.array(pil_img).astype(np.float32)
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
                elif img.ndim == 3:
                    img = np.transpose(img, (2, 0, 1))
                    
    elif suffix in ['.png', '.jpg', '.jpeg']:
        # Standard image formats
        pil_img = Image.open(image_path)
        img = np.array(pil_img).astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # Grayscale
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Ensure correct number of channels
    if img.shape[0] != expected_channels:
        img = adjust_channels(img, expected_channels)
    
    # Resize if needed
    if img.shape[1:] != target_size:
        img = resize_image(img, target_size)
    
    return img


def adjust_channels(img, target_channels):
    """
    Adjust image to have the correct number of channels.
    
    For RGB images (3 channels) being converted to 9 channels:
    - CRITICAL: NIR must be significantly BRIGHTER than SWIR for vegetation,
      otherwise NDVI/NBR collapse to 0 and no change is detected.
    - In real Sentinel-2 data, healthy vegetation has high NIR and low SWIR.
    """
    current_channels = img.shape[0]
    
    if current_channels == target_channels:
        return img
    
    if current_channels == 3 and target_channels == 9:
        # RGB to 9-channel approximation
        # Band order: B02, B03, B04, B05, B06, B07, B11, B12, B8A
        r, g, b = img[0], img[1], img[2]
        
        # Key insight: green pixels indicate vegetation → high NIR, low SWIR
        # Green channel dominance is a strong proxy for vegetation presence.
        green_frac = g / (r + g + b + 1e-8)  # 0–1, high = vegetated
        
        b02 = b                                    # Blue
        b03 = g                                    # Green
        b04 = r                                    # Red
        b05 = r * 0.6 + g * 0.4                    # Red Edge 1
        b06 = r * 0.4 + g * 0.6                    # Red Edge 2
        b07 = r * 0.3 + g * 0.7                    # Red Edge 3
        # SWIR bands: low for vegetation, moderate for bare soil
        b11 = r * 0.5 + b * 0.3 + g * 0.2          # SWIR1 — modest values
        b12 = r * 0.6 + b * 0.4                    # SWIR2 — lowest for vegetation
        # NIR: MUCH higher for vegetation (green pixels → high NIR)
        b8a = g * 1.8 + r * 0.3                    # NIR — dominated by green
        
        img = np.stack([b02, b03, b04, b05, b06, b07, b11, b12, b8a], axis=0)
        
        print(f"  ⚠️  Converted 3-channel RGB to 9-channel approximation")
        print(f"     Note: For best results, use actual Sentinel-2 multispectral data")
        
    elif current_channels < target_channels:
        repeats = target_channels // current_channels + 1
        img = np.tile(img, (repeats, 1, 1))[:target_channels]
        print(f"  ⚠️  Replicated {current_channels} channels to {target_channels}")
        
    elif current_channels > target_channels:
        img = img[:target_channels]
        print(f"  ⚠️  Truncated {current_channels} channels to {target_channels}")
    
    return img


def resize_image(img, target_size):
    """Resize image to target size using interpolation."""
    from scipy.ndimage import zoom
    
    current_h, current_w = img.shape[1], img.shape[2]
    target_h, target_w = target_size
    
    zoom_factors = (1, target_h / current_h, target_w / current_w)
    img = zoom(img, zoom_factors, order=1)
    
    print(f"  📐 Resized from ({current_h}, {current_w}) to {target_size}")
    return img


def preprocess_image(img, scale_method='clamp_scale_10000'):
    """
    Apply preprocessing to match training data distribution.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image of shape (C, H, W)
    scale_method : str
        Scaling method to apply
    
    Returns:
    --------
    torch.Tensor : Preprocessed image tensor
    """
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float()
    
    # Apply scaling based on method
    if scale_method.startswith('clamp_scale'):
        thresh = int(scale_method.split('_')[-1])
        img_tensor = torch.clamp(img_tensor, max=thresh)
        img_tensor = img_tensor / thresh
    elif scale_method == 'min-max':
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def load_model(config_path='configs/config_eval_synthetic.json'):
    """Load the pre-trained BAM-CD model."""
    
    # Load configs
    configs = pyjson5.load(open(config_path, 'r'))
    method_config_path = Path('configs') / 'method' / f"{configs['method']}.json"
    model_configs = pyjson5.load(open(method_config_path, 'r'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    model_path = configs['paths']['load_state']
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    num_channels = 9  # Sentinel-2 20m resolution
    model = init_model(configs, model_configs, checkpoint, num_channels, device)
    model = model[0]  # Extract model from tuple
    model.eval()
    
    return model, device, configs


def predict(model, device, before_img, after_img):
    """
    Run prediction on before/after image pair.
    
    Returns:
    --------
    numpy.ndarray : Binary prediction mask (0 = no change, 1 = burnt area)
    numpy.ndarray : Probability map for burnt class
    """
    with torch.no_grad():
        before_tensor = before_img.to(device)
        after_tensor = after_img.to(device)
        
        # Forward pass
        output = model(before_tensor, after_tensor)
        
        # Get predictions
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        
        # Convert to numpy
        pred_mask = pred.cpu().numpy()[0]
        prob_map = probs[0, 1].cpu().numpy()  # Probability of class 1 (burnt)
    
    return pred_mask, prob_map


def visualize_results(before_img, after_img, pred_mask, prob_map, output_path=None,
                      show=True, event_type='wildfire', hazard_result=None):
    """
    Create comprehensive visualization with color-coded severity map,
    detailed pixel-wise statistics, and proper legends.
    """
    # ── helpers ──────────────────────────────────────────────────────
    def to_rgb(img, bands=[2, 1, 0]):
        if img.shape[0] >= 3:
            rgb = img[bands].transpose(1, 2, 0)
        else:
            rgb = np.stack([img[0]] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)

    def to_false_color(img, bands=[8, 2, 1]):
        if img.shape[0] >= 9:
            fc = img[bands].transpose(1, 2, 0)
        else:
            return to_rgb(img)
        fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-8)
        return np.clip(fc, 0, 1)

    # ── event-specific config ──────────────────────────────────────
    if event_type == 'drought':
        severity_key = 'stress_severity'
        index_key    = 'ndvi_change'
        sev_labels   = ['None', 'Mild', 'Moderate', 'Severe', 'Extreme']
        sev_colors   = ['#2d6a4f', '#a7c957', '#f4a261', '#e76f51', '#9b2226']
        prob_cmap    = 'YlOrBr'
        title_prefix = '🏜️  Drought Impact Detection'
        area_label   = 'Drought'
        index_cmap   = 'BrBG'
        index_label  = 'ΔNDVI / Greenness Change'
        overlay_color = np.array([1.0, 0.5, 0.0])
    else:
        severity_key = 'burn_severity'
        index_key    = 'dnbr'
        sev_labels   = ['Unburned', 'Low', 'Moderate-Low', 'Moderate-High', 'High']
        sev_colors   = ['#2d6a4f', '#a7c957', '#f4a261', '#e76f51', '#9b2226']
        prob_cmap    = 'RdYlGn_r'
        title_prefix = '🔥  Wildfire Impact Detection'
        area_label   = 'Burnt'
        index_cmap   = 'RdYlGn'
        index_label  = 'dNBR / Burn Index'
        overlay_color = np.array([1.0, 0.0, 0.0])

    # ── extract severity map ──
    severity_map = None
    index_map    = None
    method_name  = 'unknown'
    if hazard_result is not None and hasattr(hazard_result, 'metadata'):
        severity_map = hazard_result.metadata.get(severity_key)
        index_map    = hazard_result.metadata.get(index_key)
        method_name  = hazard_result.metadata.get('method', 'unknown')

    # ── figure layout: 3 rows × 3 cols + stats panel ──────────────
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[1, 1, 1, 0.55])

    # === Row 0: Before / After / Change Magnitude ===
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after  = fig.add_subplot(gs[0, 1])
    ax_change = fig.add_subplot(gs[0, 2])

    ax_before.imshow(to_rgb(before_img))
    ax_before.set_title('Before (RGB)', fontsize=13, fontweight='bold')
    ax_before.axis('off')

    ax_after.imshow(to_rgb(after_img))
    ax_after.set_title('After (RGB)', fontsize=13, fontweight='bold')
    ax_after.axis('off')

    diff = np.abs(after_img - before_img).mean(axis=0)
    im_change = ax_change.imshow(diff, cmap='hot')
    ax_change.set_title('Change Magnitude', fontsize=13, fontweight='bold')
    ax_change.axis('off')
    plt.colorbar(im_change, ax=ax_change, fraction=0.046, pad=0.04)

    # === Row 1: False Color / Probability Map / Overlay ===
    ax_fc   = fig.add_subplot(gs[1, 0])
    ax_prob = fig.add_subplot(gs[1, 1])
    ax_over = fig.add_subplot(gs[1, 2])

    ax_fc.imshow(to_false_color(after_img))
    ax_fc.set_title('After (False Color NIR-R-G)', fontsize=13, fontweight='bold')
    ax_fc.axis('off')

    im_prob = ax_prob.imshow(prob_map, cmap=prob_cmap, vmin=0, vmax=1)
    ax_prob.set_title(f'{area_label} Probability', fontsize=13, fontweight='bold')
    ax_prob.axis('off')
    plt.colorbar(im_prob, ax=ax_prob, fraction=0.046, pad=0.04)

    # Transparent overlay on after image
    overlay = to_rgb(after_img).copy()
    affected_mask = pred_mask == 1
    alpha = 0.55
    for c in range(3):
        overlay[:, :, c] = np.where(affected_mask,
                                    overlay[:, :, c] * (1 - alpha) + overlay_color[c] * alpha,
                                    overlay[:, :, c])
    ax_over.imshow(overlay)
    ax_over.set_title(f'Detected {area_label} Areas', fontsize=13, fontweight='bold')
    ax_over.axis('off')
    # Add legend patches
    legend_patches = [
        mpatches.Patch(color=overlay_color, label=f'{area_label} detected', alpha=0.7),
        mpatches.Patch(color='gray', label='Unaffected', alpha=0.5),
    ]
    ax_over.legend(handles=legend_patches, loc='lower right', fontsize=9,
                   framealpha=0.8, edgecolor='black')

    # === Row 2: Index map / Color-coded severity / Index histogram ===
    ax_idx  = fig.add_subplot(gs[2, 0])
    ax_sev  = fig.add_subplot(gs[2, 1])
    ax_hist = fig.add_subplot(gs[2, 2])

    # Spectral index map
    if index_map is not None:
        vabs = max(abs(np.nanpercentile(index_map, 2)),
                   abs(np.nanpercentile(index_map, 98)), 0.01)
        im_idx = ax_idx.imshow(index_map, cmap=index_cmap, vmin=-vabs, vmax=vabs)
        ax_idx.set_title(index_label, fontsize=13, fontweight='bold')
        plt.colorbar(im_idx, ax=ax_idx, fraction=0.046, pad=0.04)
    else:
        ax_idx.imshow(diff, cmap='hot')
        ax_idx.set_title('Spectral Change (proxy)', fontsize=13, fontweight='bold')
    ax_idx.axis('off')

    # Color-coded severity map with legend
    if severity_map is not None:
        cmap_sev = ListedColormap(sev_colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm_sev = BoundaryNorm(bounds, cmap_sev.N)
        im_sev = ax_sev.imshow(severity_map, cmap=cmap_sev, norm=norm_sev, interpolation='nearest')
        ax_sev.set_title(f'{area_label} Severity (per pixel)', fontsize=13, fontweight='bold')
        ax_sev.axis('off')
        # Create legend
        sev_patches = [mpatches.Patch(color=sev_colors[i], label=sev_labels[i])
                       for i in range(len(sev_labels))]
        ax_sev.legend(handles=sev_patches, loc='lower right', fontsize=8,
                      framealpha=0.9, edgecolor='black', title='Severity',
                      title_fontsize=9)
    else:
        ax_sev.imshow(prob_map, cmap='hot')
        ax_sev.set_title('Probability Heatmap', fontsize=13, fontweight='bold')
        ax_sev.axis('off')

    # Probability histogram
    ax_hist.hist(prob_map.ravel(), bins=50, color='#e76f51', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_hist.set_title(f'{area_label} Probability Distribution', fontsize=13, fontweight='bold')
    ax_hist.set_xlabel('Probability', fontsize=11)
    ax_hist.set_ylabel('Pixel Count', fontsize=11)
    ax_hist.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(axis='y', alpha=0.3)

    # === Row 3: Statistics panel (spanning all 3 cols) ===
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')

    total_pixels = pred_mask.size
    affected_pixels = int(affected_mask.sum())
    unaffected_pixels = total_pixels - affected_pixels
    affected_pct = (affected_pixels / total_pixels) * 100
    unaffected_pct = 100.0 - affected_pct

    # Build statistics table data
    table_data = [
        ['Total Pixels', f'{total_pixels:,}', '100.00%'],
        [f'{area_label} Pixels', f'{affected_pixels:,}', f'{affected_pct:.2f}%'],
        ['Unaffected Pixels', f'{unaffected_pixels:,}', f'{unaffected_pct:.2f}%'],
        ['', '', ''],
        ['Mean Probability', f'{prob_map.mean():.4f}', ''],
        ['Max Probability', f'{prob_map.max():.4f}', ''],
        ['Median Probability', f'{np.median(prob_map):.4f}', ''],
        ['Detection Method', method_name, ''],
    ]

    # Add per-severity breakdown
    if severity_map is not None:
        table_data.append(['', '', ''])
        table_data.append(['── SEVERITY BREAKDOWN ──', '', ''])
        for code, label in enumerate(sev_labels):
            count = int((severity_map == code).sum())
            pct = (count / total_pixels) * 100
            table_data.append([f'  {label}', f'{count:,} px', f'{pct:.2f}%'])

    # Add confidence
    if hazard_result is not None:
        table_data.append(['', '', ''])
        table_data.append(['Model Confidence', f'{hazard_result.confidence:.2f}', ''])

    tbl = ax_stats.table(
        cellText=table_data,
        colLabels=['Metric', 'Value', 'Percentage'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.3, 0.2],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)

    # Style the table
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#264653')
            cell.set_text_props(color='white', fontweight='bold')
        elif row >= 1 and table_data[row - 1][0].startswith('──'):
            cell.set_facecolor('#e9ecef')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
        cell.set_edgecolor('#dee2e6')

    fig.suptitle(
        f'{title_prefix}\n'
        f'{area_label} Area: {affected_pixels:,} pixels ({affected_pct:.2f}%) '
        f'| Total: {total_pixels:,} pixels',
        fontsize=16, fontweight='bold', y=0.98
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Visualization saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def run_demo(event_type='wildfire'):
    """Run prediction on sample data to demonstrate the pipeline."""
    event_emoji = '🏜️' if event_type == 'drought' else '🔥'
    event_name = 'Drought' if event_type == 'drought' else 'Wildfire'
    print(f"\n{font_colors.BOLD}{event_emoji} Running {event_name} Demo with Synthetic Data{font_colors.ENDC}\n")
    
    # Load sample images
    data_dir = Path('data/processed/2024')
    before_path = data_dir / 'synthetic_event_test_000_patch_000.S2_before.npy'
    after_path = data_dir / 'synthetic_event_test_000_patch_000.S2_after.npy'
    label_path = data_dir / 'synthetic_event_test_000_patch_000.label.npy'
    
    if not before_path.exists():
        print(f"{font_colors.RED}Demo data not found. Please ensure synthetic dataset exists.{font_colors.ENDC}")
        return
    
    # Load and predict
    print("Loading images...")
    before_img = np.load(before_path).astype(np.float32)
    after_img = np.load(after_path).astype(np.float32)
    label = np.load(label_path)
    
    print("Loading model...")
    model, device, configs = load_model()
    
    print("Preprocessing...")
    before_tensor = preprocess_image(before_img, configs['datasets']['scale_input'])
    after_tensor = preprocess_image(after_img, configs['datasets']['scale_input'])
    
    print("Running prediction...")
    pred_mask, prob_map = predict(model, device, before_tensor, after_tensor)
    
    # Calculate metrics
    from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
    iou_metric = BinaryJaccardIndex()
    f1_metric = BinaryF1Score()
    
    pred_t = torch.from_numpy(pred_mask).long()
    label_t = torch.from_numpy(label).long()
    
    iou = iou_metric(pred_t, label_t).item()
    f1 = f1_metric(pred_t, label_t).item()
    
    print(f"\n{font_colors.GREEN}Demo Results:{font_colors.ENDC}")
    print(f"  IoU Score: {iou:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Visualize
    output_path = Path('results/demo_prediction.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    visualize_results(before_img, after_img, pred_mask, prob_map, output_path=str(output_path), event_type=event_type)
    
    return pred_mask, prob_map


def main():
    parser = argparse.ArgumentParser(
        description='Wildfire Impact Detection from Satellite Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--before', type=str, help='Path to before-fire satellite image')
    parser.add_argument('--after', type=str, help='Path to after-fire satellite image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output visualization')
    parser.add_argument('--output-mask', type=str, default=None, help='Path to save binary prediction mask (.npy)')
    parser.add_argument('--output-prob', type=str, default=None, help='Path to save probability map (.npy)')
    parser.add_argument('--config', type=str, default='configs/config_eval_synthetic.json', 
                        help='Path to config file')
    parser.add_argument('--rgb', action='store_true', 
                        help='Input images are RGB (will be converted to 9-channel approximation)')
    parser.add_argument('--bands', type=int, default=9, 
                        help='Number of bands in input images (default: 9)')
    parser.add_argument('--no-show', action='store_true', 
                        help='Do not display visualization (useful for batch processing)')
    parser.add_argument('--demo', action='store_true', 
                        help='Run demo with sample data')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary prediction (default: 0.5)')
    parser.add_argument('--drought', action='store_true',
                        help='Run in drought detection mode instead of wildfire')
    parser.add_argument('--hazard', type=str, choices=['wildfire', 'drought', 'rainfall', 'vegetation'],
                        help='Specify the hazard type to analyze (overrides --drought)')
    
    args = parser.parse_args()
    
    # Determine event type
    if args.hazard:
        event_type = args.hazard
    else:
        event_type = 'drought' if args.drought else 'wildfire'
        
    if event_type == 'drought':
        event_emoji = '🏜️'
        event_name = 'Drought'
    elif event_type == 'rainfall':
        event_emoji = '🌧️'
        event_name = 'Rainfall / Flood'
    elif event_type == 'vegetation':
        event_emoji = '🌿'
        event_name = 'Vegetation Health'
    else:
        event_type = 'wildfire'
        event_emoji = '🔥'
        event_name = 'Wildfire'
    
    print(f"\n{font_colors.BOLD}{'='*60}")
    print(f"{event_emoji} {event_name} Impact Detection from Satellite Images")
    print(f"{'='*60}{font_colors.ENDC}\n")
    
    # Run demo if requested
    if args.demo:
        run_demo(event_type=event_type)
        return
    
    # Check required arguments
    if not args.before or not args.after:
        parser.error("--before and --after are required (or use --demo for demo mode)")
    
    # Check if ML model is supported for this hazard
    use_ml_model = event_type in ['wildfire', 'drought']
    
    # Load model with appropriate config if applicable
    if use_ml_model:
        config_path = args.config
        if event_type == 'drought' and args.config == 'configs/config_eval_synthetic.json':
            config_path = 'configs/config_eval_synthetic_drought.json'
        
        print("📦 Loading pre-trained model...")
        model, device, configs = load_model(config_path)
        print(f"   Device: {device}")
        print(f"   Model: BAM-CD")
        print(f"   Mode: {event_name} Detection")
        scale_method = configs['datasets']['scale_input']
    else:
        print(f"📦 Non-ML Mode: {event_name} Detection")
        scale_method = 'min-max'  # Default for non-ML
    
    # Load images
    print(f"\n📷 Loading images...")
    expected_channels = 3 if args.rgb else args.bands
    
    print(f"   Before: {args.before}")
    before_img = load_image(args.before, expected_channels=expected_channels)
    
    print(f"   After:  {args.after}")
    after_img = load_image(args.after, expected_channels=expected_channels)
    
    # Adjust channels if RGB mode
    if args.rgb:
        before_img = adjust_channels(before_img, 9)
        after_img = adjust_channels(after_img, 9)
    
    print(f"   Image shape: {before_img.shape}")
    
    # Preprocess
    print(f"\n⚙️  Preprocessing...")
    print(f"   Scaling method: {scale_method}")
    
    before_tensor = preprocess_image(before_img, scale_method)
    after_tensor = preprocess_image(after_img, scale_method)
    
    if use_ml_model:
        # Predict with ML model
        print(f"\n🔮 Running ML model prediction...")
        pred_mask, prob_map = predict(model, device, before_tensor, after_tensor)
        
        # Apply custom threshold if specified
        if args.threshold != 0.5:
            pred_mask = (prob_map >= args.threshold).astype(np.int64)
            print(f"   Applied threshold: {args.threshold}")
    else:
        pred_mask = np.zeros((before_img.shape[1], before_img.shape[2]), dtype=np.int64)
        prob_map = np.zeros((before_img.shape[1], before_img.shape[2]), dtype=np.float32)
    
    # Also run hazard module analysis (index-based) for detailed severity info
    print(f"\n🔬 Running spectral index analysis...")
    hazard_result = None
    try:
        if event_type == 'drought':
            from hazard_drought import DroughtModule
            hmod = DroughtModule()
        elif event_type == 'rainfall':
            from hazard_rainfall import RainfallModule
            hmod = RainfallModule()
        elif event_type == 'vegetation':
            from hazard_vegetation import VegetationModule
            hmod = VegetationModule()
        else:
            from hazard_wildfire import WildfireModule
            hmod = WildfireModule()

        hazard_result = hmod.analyze(before_img, after_img)

        # Use hazard module results if ML model produced nothing meaningful
        ml_affected = int((pred_mask == 1).sum())
        idx_affected = int((hazard_result.binary_mask == 1).sum())
        
        if not use_ml_model:
            print(f"   ℹ️  Using index-based analysis ({idx_affected:,} pixels)")
            pred_mask = hazard_result.binary_mask.astype(pred_mask.dtype)
            prob_map  = hazard_result.probability_map
        elif ml_affected == 0 and idx_affected > 0:
            print(f"   ℹ️  ML model detected 0 pixels; using index-based analysis ({idx_affected:,} pixels)")
            pred_mask = hazard_result.binary_mask.astype(pred_mask.dtype)
            prob_map  = hazard_result.probability_map
        elif idx_affected > ml_affected:
            print(f"   ℹ️  Combining ML ({ml_affected:,} px) + index ({idx_affected:,} px) results")
            pred_mask = np.maximum(pred_mask, hazard_result.binary_mask.astype(pred_mask.dtype))
            prob_map  = np.maximum(prob_map, hazard_result.probability_map)
    except Exception as e:
        print(f"   ⚠️  Hazard module analysis skipped: {e}")
    
    # Calculate statistics
    total_pixels = pred_mask.size
    affected_pixels = int((pred_mask == 1).sum())
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    area_label = 'Drought' if args.drought else 'Burnt'
    prob_label = 'drought' if args.drought else 'burn'
    
    print(f"\n{font_colors.GREEN}📊 Results:{font_colors.ENDC}")
    print(f"   Total pixels:     {total_pixels:,}")
    print(f"   {area_label} pixels:  {affected_pixels:,}")
    print(f"   {area_label} area:    {affected_percentage:.2f}%")
    print(f"   Avg {prob_label} prob: {prob_map.mean():.4f}")
    print(f"   Max {prob_label} prob: {prob_map.max():.4f}")
    if hazard_result is not None:
        sev_key = 'stress_severity' if args.drought else 'burn_severity'
        sev_labels = ['None','Mild','Moderate','Severe','Extreme'] if args.drought else \
                     ['Unburned','Low','Mod-Low','Mod-High','High']
        sev_map = hazard_result.metadata.get(sev_key)
        if sev_map is not None:
            print(f"\n   Severity breakdown:")
            for code, lbl in enumerate(sev_labels):
                cnt = int((sev_map == code).sum())
                pct = cnt / total_pixels * 100
                print(f"     {lbl:>12s}: {cnt:>8,} px  ({pct:5.2f}%)")
        print(f"   Method: {hazard_result.metadata.get('method', 'unknown')}")
        print(f"   Confidence: {hazard_result.confidence:.2f}")
    
    # Save outputs if requested
    if args.output_mask:
        np.save(args.output_mask, pred_mask)
        print(f"\n💾 Saved prediction mask to: {args.output_mask}")
    
    if args.output_prob:
        np.save(args.output_prob, prob_map)
        print(f"💾 Saved probability map to: {args.output_prob}")
    
    # Visualize
    print(f"\n🎨 Generating visualization...")
    output_path = args.output if args.output else 'results/prediction_output.png'
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    visualize_results(
        before_img, after_img, pred_mask, prob_map,
        output_path=output_path,
        show=not args.no_show,
        event_type=event_type,
        hazard_result=hazard_result,
    )
    
    print(f"\n{font_colors.GREEN}✅ Done!{font_colors.ENDC}\n")
    
    return pred_mask, prob_map


if __name__ == '__main__':
    main()
