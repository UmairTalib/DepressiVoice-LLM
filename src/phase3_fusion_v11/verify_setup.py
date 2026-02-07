
import torch
from models import FusionV11Classifier

def verify_model():
    print("Verifying FusionV11Classifier with Dummy Data...")
    
    # Dimensions
    B, T = 4, 32
    audio_dim = 1024
    text_dim = 768
    
    # 1. Create Dummy Data
    audio = torch.randn(B, T, audio_dim)
    text = torch.randn(B, T, text_dim)
    scalars = torch.randn(B, 3)
    
    # Create masks (some padding)
    # Let's say last 2 steps are padded for 1st sample
    audio_mask = torch.zeros(B, T, dtype=torch.bool)
    audio_mask[0, -2:] = True
    
    text_mask = torch.zeros(B, T, dtype=torch.bool)
    text_mask[0, -2:] = True
    
    # 2. Init Model
    model = FusionV11Classifier(
        audio_dim=audio_dim,
        text_dim=text_dim,
        shared_dim=256,
        nhead=4,
        dropout=0.1
    )
    
    # 3. Forward Pass
    print("Running Forward Pass...")
    logits, z = model(audio, text, scalars, audio_mask, text_mask)
    
    # 4. Check Outputs
    print(f"Logits Shape: {logits.shape} (Expected [{B}, 1])")
    print(f"Gate z Shape: {z.shape} (Expected [{B}, {T}, 256])")
    
    assert logits.shape == (B, 1)
    # z shape is [Batch, T, Shared] because it's per-step gate
    assert z.shape == (B, T, 256)
    
    print("\n✅ Verification Successful: Model forward pass works correctly.")

if __name__ == "__main__":
    verify_model()
