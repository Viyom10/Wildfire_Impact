"""
Wildfire Impact Detection - Web Interface

A user-friendly web interface for uploading satellite images and detecting wildfire burn areas.
Uses Gradio for the interface.

To run:
    python web_interface.py

Then open your browser to http://localhost:7860
"""

import numpy as np
from pathlib import Path
import torch
import pyjson5
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import io
import base64

warnings.filterwarnings('ignore')

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Install with: pip install gradio")

from utils import init_model, font_colors


# Global model cache
MODEL_CACHE = {
    'model': None,
    'device': None,
    'configs': None
}


def load_model_cached():
    """Load model once and cache it."""
    if MODEL_CACHE['model'] is None:
        config_path = 'configs/config_eval_synthetic.json'
        configs = pyjson5.load(open(config_path, 'r'))
        method_config_path = Path('configs') / 'method' / f"{configs['method']}.json"
        model_configs = pyjson5.load(open(method_config_path, 'r'))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = configs['paths']['load_state']
        checkpoint = torch.load(model_path, map_location=device)
        
        num_channels = 9
        model = init_model(configs, model_configs, checkpoint, num_channels, device)
        model = model[0]
        model.eval()
        
        MODEL_CACHE['model'] = model
        MODEL_CACHE['device'] = device
        MODEL_CACHE['configs'] = configs
        
    return MODEL_CACHE['model'], MODEL_CACHE['device'], MODEL_CACHE['configs']


def process_uploaded_image(image, expected_channels=9, target_size=(256, 256)):
    """
    Process an uploaded image into the format expected by the model.
    
    Parameters:
    -----------
    image : PIL.Image or numpy.ndarray
        Uploaded image
    expected_channels : int
        Number of channels expected (9 for Sentinel-2)
    target_size : tuple
        Target (height, width)
    
    Returns:
    --------
    numpy.ndarray : Processed image (C, H, W)
    """
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        img = np.array(image).astype(np.float32)
    else:
        img = image.astype(np.float32)
    
    # Handle different dimensions
    if img.ndim == 2:
        # Grayscale
        img = img[np.newaxis, :, :]
    elif img.ndim == 3:
        if img.shape[2] <= 12:  # Assume (H, W, C)
            img = np.transpose(img, (2, 0, 1))
    
    # Adjust channels
    current_channels = img.shape[0]
    if current_channels != expected_channels:
        if current_channels == 3:
            # RGB to 9-channel approximation
            r, g, b = img[0], img[1], img[2]
            b02 = b
            b03 = g
            b04 = r
            b05 = (r + g) / 2
            b06 = (r * 0.7 + g * 0.3)
            b07 = r * 0.8 + g * 0.2
            b11 = (r + b) / 2
            b12 = r * 0.6 + b * 0.4
            b8a = (r + g + b) / 3
            img = np.stack([b02, b03, b04, b05, b06, b07, b11, b12, b8a], axis=0)
        elif current_channels == 4:
            # RGBA - drop alpha, convert RGB
            img = img[:3]
            return process_uploaded_image(img.transpose(1, 2, 0), expected_channels, target_size)
        else:
            # Replicate or truncate
            if current_channels < expected_channels:
                repeats = expected_channels // current_channels + 1
                img = np.tile(img, (repeats, 1, 1))[:expected_channels]
            else:
                img = img[:expected_channels]
    
    # Resize if needed
    if img.shape[1:] != target_size:
        from scipy.ndimage import zoom
        current_h, current_w = img.shape[1], img.shape[2]
        target_h, target_w = target_size
        zoom_factors = (1, target_h / current_h, target_w / current_w)
        img = zoom(img, zoom_factors, order=1)
    
    return img


def preprocess_for_model(img, scale_method='clamp_scale_10000'):
    """Apply preprocessing."""
    img_tensor = torch.from_numpy(img).float()
    
    if scale_method.startswith('clamp_scale'):
        thresh = int(scale_method.split('_')[-1])
        img_tensor = torch.clamp(img_tensor, max=thresh)
        img_tensor = img_tensor / thresh
    
    return img_tensor.unsqueeze(0)


def create_visualization(before_img, after_img, pred_mask, prob_map):
    """Create a visualization figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    def to_rgb(img, bands=[2, 1, 0]):
        if img.shape[0] >= 3:
            rgb = img[bands].transpose(1, 2, 0)
        else:
            rgb = np.stack([img[0]] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)
    
    # Row 1
    axes[0, 0].imshow(to_rgb(before_img))
    axes[0, 0].set_title('Before Image', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_rgb(after_img))
    axes[0, 1].set_title('After Image', fontsize=11)
    axes[0, 1].axis('off')
    
    diff = np.abs(after_img - before_img).mean(axis=0)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Change Magnitude', fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2
    # False color
    if before_img.shape[0] >= 9:
        fc = before_img[[8, 2, 1]].transpose(1, 2, 0)
        fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-8)
        axes[1, 0].imshow(np.clip(fc, 0, 1))
        axes[1, 0].set_title('Before (False Color)', fontsize=11)
    else:
        axes[1, 0].imshow(to_rgb(before_img))
        axes[1, 0].set_title('Before (RGB)', fontsize=11)
    axes[1, 0].axis('off')
    
    # Probability map
    im = axes[1, 1].imshow(prob_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 1].set_title('Burn Probability', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = to_rgb(after_img).copy()
    burn_mask = pred_mask == 1
    overlay[burn_mask] = [1, 0, 0]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Detected Burn Areas (Red)', fontsize=11)
    axes[1, 2].axis('off')
    
    # Stats
    total_pixels = pred_mask.size
    burnt_pixels = burn_mask.sum()
    burnt_percentage = (burnt_pixels / total_pixels) * 100
    
    fig.suptitle(f'🔥 Wildfire Impact Detection\nBurnt Area: {burnt_pixels:,} pixels ({burnt_percentage:.2f}%)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def predict_burn_areas(before_image, after_image, threshold=0.5, progress=gr.Progress()):
    """
    Main prediction function for Gradio interface.
    
    Parameters:
    -----------
    before_image : PIL.Image
        Image before the fire
    after_image : PIL.Image
        Image after the fire
    threshold : float
        Probability threshold for classification
    
    Returns:
    --------
    tuple : (visualization figure, statistics text, prediction mask image)
    """
    if before_image is None or after_image is None:
        return None, "Please upload both before and after images.", None
    
    try:
        progress(0.1, desc="Loading model...")
        model, device, configs = load_model_cached()
        
        progress(0.3, desc="Processing images...")
        before_img = process_uploaded_image(before_image)
        after_img = process_uploaded_image(after_image)
        
        progress(0.5, desc="Preprocessing...")
        scale_method = configs['datasets']['scale_input']
        before_tensor = preprocess_for_model(before_img, scale_method)
        after_tensor = preprocess_for_model(after_img, scale_method)
        
        progress(0.7, desc="Running prediction...")
        with torch.no_grad():
            before_t = before_tensor.to(device)
            after_t = after_tensor.to(device)
            output = model(before_t, after_t)
            probs = torch.softmax(output, dim=1)
            pred_mask = (probs[0, 1] >= threshold).cpu().numpy().astype(np.uint8)
            prob_map = probs[0, 1].cpu().numpy()
        
        progress(0.9, desc="Creating visualization...")
        
        # Statistics
        total_pixels = pred_mask.size
        burnt_pixels = pred_mask.sum()
        burnt_percentage = (burnt_pixels / total_pixels) * 100
        
        stats = f"""
## 📊 Detection Results

| Metric | Value |
|--------|-------|
| **Total Pixels** | {total_pixels:,} |
| **Burnt Pixels** | {burnt_pixels:,} |
| **Burnt Area** | {burnt_percentage:.2f}% |
| **Avg Burn Probability** | {prob_map.mean():.4f} |
| **Max Burn Probability** | {prob_map.max():.4f} |
| **Threshold Used** | {threshold} |

### Interpretation
- 🔴 **Red areas** in the overlay indicate detected burn zones
- Higher probability values indicate higher confidence
- Adjust threshold to be more/less sensitive
"""
        
        # Create visualization
        fig = create_visualization(before_img, after_img, pred_mask, prob_map)
        
        # Create mask image for download
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        
        # Create probability heatmap image
        prob_img = plt.cm.RdYlGn_r(prob_map)[:, :, :3]
        prob_img = Image.fromarray((prob_img * 255).astype(np.uint8))
        
        progress(1.0, desc="Done!")
        
        return fig, stats, mask_img, prob_img
        
    except Exception as e:
        import traceback
        error_msg = f"Error during prediction: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None, None


def load_sample_images():
    """Load sample images for demo."""
    data_dir = Path('data/processed/2024')
    before_path = data_dir / 'synthetic_event_test_000_patch_000.S2_before.npy'
    after_path = data_dir / 'synthetic_event_test_000_patch_000.S2_after.npy'
    
    if before_path.exists() and after_path.exists():
        before_img = np.load(before_path).astype(np.float32)
        after_img = np.load(after_path).astype(np.float32)
        
        # Convert to RGB-like for display
        def to_pil(img):
            rgb = img[[2, 1, 0]].transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
            return Image.fromarray(rgb.astype(np.uint8))
        
        return to_pil(before_img), to_pil(after_img)
    
    return None, None


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="🔥 Wildfire Impact Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔥 Wildfire Impact Detection from Satellite Images
        
        Upload **before** and **after** satellite images to detect burn areas caused by wildfires.
        
        ### Instructions:
        1. Upload a satellite image taken **before** the fire event
        2. Upload a satellite image taken **after** the fire event
        3. Adjust the detection threshold if needed
        4. Click "Detect Burn Areas" to run the analysis
        
        ### Supported Formats:
        - **GeoTIFF** (.tif, .tiff) - Best for multi-band satellite data
        - **Standard images** (.png, .jpg) - Will be converted to approximate spectral bands
        - **NumPy arrays** (.npy) - Direct array input
        
        ⚠️ **Note**: For best results, use actual Sentinel-2 multispectral imagery with 9 bands.
        RGB images will be converted to approximate spectral bands.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                before_input = gr.Image(
                    label="📷 Before Image", 
                    type="pil",
                    height=300
                )
                
            with gr.Column(scale=1):
                after_input = gr.Image(
                    label="📷 After Image", 
                    type="pil",
                    height=300
                )
        
        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.9, 
                value=0.5, 
                step=0.05,
                label="Detection Threshold",
                info="Lower = more sensitive, Higher = more conservative"
            )
            
            detect_btn = gr.Button("🔍 Detect Burn Areas", variant="primary", size="lg")
            demo_btn = gr.Button("📂 Load Demo Images", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Detection Results")
            
            with gr.Column(scale=1):
                output_stats = gr.Markdown(label="Statistics")
        
        with gr.Row():
            with gr.Column():
                output_mask = gr.Image(label="🎭 Binary Mask (Download)", type="pil")
            with gr.Column():
                output_prob = gr.Image(label="🌡️ Probability Heatmap (Download)", type="pil")
        
        # Event handlers
        detect_btn.click(
            fn=predict_burn_areas,
            inputs=[before_input, after_input, threshold_slider],
            outputs=[output_plot, output_stats, output_mask, output_prob]
        )
        
        def load_demo():
            before, after = load_sample_images()
            if before is None:
                gr.Warning("Demo images not found. Please use your own images.")
                return None, None
            return before, after
        
        demo_btn.click(
            fn=load_demo,
            outputs=[before_input, after_input]
        )
        
        gr.Markdown("""
        ---
        ### About the Model
        
        This system uses a **BAM-CD (Bitemporal Attention Module Change Detection)** model 
        trained on Sentinel-2 satellite imagery to detect burn areas from wildfires.
        
        **Model Input**: 9-band Sentinel-2 imagery at 20m resolution  
        **Bands**: B02, B03, B04, B05, B06, B07, B11, B12, B8A
        
        **Output**: Binary burn mask and probability map
        """)
    
    return demo


def main():
    if not GRADIO_AVAILABLE:
        print(f"\n{font_colors.RED}Gradio is not installed!{font_colors.ENDC}")
        print("Install it with: pip install gradio")
        print("\nAlternatively, use the command-line tool:")
        print("  python predict_from_images.py --before image1.tif --after image2.tif")
        return
    
    print(f"\n{font_colors.BOLD}🔥 Starting Wildfire Impact Detection Web Interface{font_colors.ENDC}\n")
    
    # Pre-load model
    print("Loading model...")
    try:
        load_model_cached()
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"⚠️ Model will load on first prediction: {e}\n")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )


if __name__ == '__main__':
    main()
