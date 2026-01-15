"""
Quick test script to verify BAM-CD model can be loaded
"""
import torch
import pyjson5
from pathlib import Path
from utils import init_model

print("=" * 60)
print("BAM-CD Model Loading Test")
print("=" * 60)

# Load configs
config = pyjson5.load(open('configs/config_eval.json'))
model_config = pyjson5.load(open('configs/method/bam_cd.json'))

# Check if pretrained model exists
model_path = Path(config['paths']['load_state'])
if not model_path.exists():
    print(f"\n❌ Model not found at: {model_path}")
    print("\nPlease download the model first:")
    print("V1: https://www.dropbox.com/scl/fo/7boya0nvjb0sgmg9l5quo/ALldNkZ5d-DMtUMmUPm5O4A?rlkey=1p29xzb912mj4hh68k8waac9x&dl=1")
    print("V2: https://www.dropbox.com/scl/fo/8bhch7e0yhwgfqxsc6i69/ANcImDZyq7EYHVPt66aYuq8?rlkey=oiz26v6r09t4bpr5fize3a9a8&dl=1")
    print(f"\nSave to: {model_path}")
    exit(1)

print(f"\n✅ Found model at: {model_path}")

# Try to load checkpoint
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    checkpoint = torch.load(model_path, map_location=device)
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Keys: {list(checkpoint.keys())}")
    
    # Initialize model
    model = init_model(config, model_config, checkpoint, 9, device)
    print(f"✅ Model initialized successfully")
    
    # Test forward pass
    x1 = torch.randn(1, 9, 256, 256).to(device)
    x2 = torch.randn(1, 9, 256, 256).to(device)
    
    model[0].eval()
    with torch.no_grad():
        output = model[0](x1, x2)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {x1.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("The model is ready to use for evaluation.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
