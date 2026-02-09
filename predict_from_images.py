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
    - The model expects specific Sentinel-2 bands
    - We replicate and transform RGB to approximate the spectral response
    """
    current_channels = img.shape[0]
    
    if current_channels == target_channels:
        return img
    
    if current_channels == 3 and target_channels == 9:
        # RGB to 9-channel approximation
        # Map RGB to approximate Sentinel-2 bands
        # B02 (Blue), B03 (Green), B04 (Red), B05-B07 (Red Edge), B11-B12 (SWIR), B8A (NIR)
        r, g, b = img[0], img[1], img[2]
        
        # Create approximate band mappings
        b02 = b  # Blue
        b03 = g  # Green
        b04 = r  # Red
        b05 = (r + g) / 2  # Red Edge 1 (approx)
        b06 = (r * 0.7 + g * 0.3)  # Red Edge 2 (approx)
        b07 = r * 0.8 + g * 0.2  # Red Edge 3 (approx)
        b11 = (r + b) / 2  # SWIR 1 (approx)
        b12 = r * 0.6 + b * 0.4  # SWIR 2 (approx)
        b8a = (r + g + b) / 3  # NIR (approx)
        
        img = np.stack([b02, b03, b04, b05, b06, b07, b11, b12, b8a], axis=0)
        
        print(f"  ⚠️  Converted 3-channel RGB to 9-channel approximation")
        print(f"     Note: For best results, use actual Sentinel-2 multispectral data")
        
    elif current_channels < target_channels:
        # Replicate channels to reach target
        repeats = target_channels // current_channels + 1
        img = np.tile(img, (repeats, 1, 1))[:target_channels]
        print(f"  ⚠️  Replicated {current_channels} channels to {target_channels}")
        
    elif current_channels > target_channels:
        # Take first N channels
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


def visualize_results(before_img, after_img, pred_mask, prob_map, output_path=None, show=True, event_type='wildfire'):
    """
    Create visualization of the prediction results.
    
    Parameters:
    -----------
    before_img : numpy.ndarray
        Before image (C, H, W)
    after_img : numpy.ndarray
        After image (C, H, W)
    pred_mask : numpy.ndarray
        Binary prediction mask
    prob_map : numpy.ndarray
        Probability map
    output_path : str, optional
        Path to save the visualization
    show : bool
        Whether to display the plot
    event_type : str
        Type of event: 'wildfire' or 'drought'
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Helper to create RGB composite
    def to_rgb(img, bands=[2, 1, 0]):
        """Convert multi-channel image to RGB for visualization."""
        if img.shape[0] >= 3:
            rgb = img[bands].transpose(1, 2, 0)
        else:
            rgb = np.stack([img[0]] * 3, axis=-1)
        
        # Normalize for display
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)
    
    # Create false color composite (NIR-R-G like bands 8A-4-3)
    def to_false_color(img, bands=[8, 2, 1]):
        """Create false color composite for better vegetation/burn visualization."""
        if img.shape[0] >= 9:
            fc = img[bands].transpose(1, 2, 0)
        else:
            return to_rgb(img)
        fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-8)
        return np.clip(fc, 0, 1)
    
    # Row 1: Before/After images
    axes[0, 0].imshow(to_rgb(before_img))
    axes[0, 0].set_title('Before (RGB)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_rgb(after_img))
    axes[0, 1].set_title('After (RGB)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Difference image
    diff = np.abs(after_img - before_img).mean(axis=0)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Change Magnitude', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: False color and predictions
    axes[1, 0].imshow(to_false_color(after_img))
    axes[1, 0].set_title('After (False Color NIR-R-G)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Set labels based on event type
    if event_type == 'drought':
        prob_label = 'Drought Probability'
        detect_label = 'Detected Drought Areas (Orange)'
        overlay_color = [1, 0.5, 0]  # Orange for drought
        title_prefix = 'Drought Impact Detection'
        area_label = 'Drought Area'
        cmap = 'YlOrBr'
    else:
        prob_label = 'Burn Probability'
        detect_label = 'Detected Burn Areas (Red)'
        overlay_color = [1, 0, 0]  # Red for burnt
        title_prefix = 'Wildfire Impact Detection'
        area_label = 'Burnt Area'
        cmap = 'RdYlGn_r'
    
    # Probability map
    im = axes[1, 1].imshow(prob_map, cmap=cmap, vmin=0, vmax=1)
    axes[1, 1].set_title(prob_label, fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Binary prediction overlay
    overlay = to_rgb(after_img).copy()
    # Create overlay for affected areas
    affected_mask = pred_mask == 1
    overlay[affected_mask] = overlay_color
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(detect_label, fontsize=12)
    axes[1, 2].axis('off')
    
    # Add statistics
    total_pixels = pred_mask.size
    affected_pixels = affected_mask.sum()
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    fig.suptitle(f'{title_prefix}\n{area_label}: {affected_pixels:,} pixels ({affected_percentage:.2f}%)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    
    args = parser.parse_args()
    
    # Determine event type
    event_type = 'drought' if args.drought else 'wildfire'
    event_emoji = '🏜️' if args.drought else '🔥'
    event_name = 'Drought' if args.drought else 'Wildfire'
    
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
    
    # Load model with appropriate config
    config_path = args.config
    if args.drought and args.config == 'configs/config_eval_synthetic.json':
        config_path = 'configs/config_eval_synthetic_drought.json'
    
    print("📦 Loading pre-trained model...")
    model, device, configs = load_model(config_path)
    print(f"   Device: {device}")
    print(f"   Model: BAM-CD")
    print(f"   Mode: {event_name} Detection")
    
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
    scale_method = configs['datasets']['scale_input']
    print(f"   Scaling method: {scale_method}")
    
    before_tensor = preprocess_image(before_img, scale_method)
    after_tensor = preprocess_image(after_img, scale_method)
    
    # Predict
    print(f"\n🔮 Running prediction...")
    pred_mask, prob_map = predict(model, device, before_tensor, after_tensor)
    
    # Apply custom threshold if specified
    if args.threshold != 0.5:
        pred_mask = (prob_map >= args.threshold).astype(np.int64)
        print(f"   Applied threshold: {args.threshold}")
    
    # Calculate statistics
    total_pixels = pred_mask.size
    affected_pixels = (pred_mask == 1).sum()
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    # Labels based on mode
    area_label = 'Drought' if args.drought else 'Burnt'
    prob_label = 'drought' if args.drought else 'burn'
    
    print(f"\n{font_colors.GREEN}📊 Results:{font_colors.ENDC}")
    print(f"   Total pixels:     {total_pixels:,}")
    print(f"   {area_label} pixels:  {affected_pixels:,}")
    print(f"   {area_label} area:    {affected_percentage:.2f}%")
    print(f"   Avg {prob_label} prob: {prob_map.mean():.4f}")
    print(f"   Max {prob_label} prob: {prob_map.max():.4f}")
    
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
        event_type=event_type
    )
    
    print(f"\n{font_colors.GREEN}✅ Done!{font_colors.ENDC}\n")
    
    return pred_mask, prob_map


if __name__ == '__main__':
    main()
