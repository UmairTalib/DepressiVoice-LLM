import torch
import os
import sys

# Add src to path
sys.path.append(os.path.expanduser('~/DepressiVoice-LLM/src/phase3_fusion_v10'))

from models import FusionV10Model
from dataset import DepressionDataset

def main():
    print("=== Interactive Verification ===", flush=True)
    
    # 1. Load Data Sample directly
    pid = "300"
    audio_path = os.path.expanduser(f'~/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/{pid}_features.pt')
    text_path = os.path.expanduser(f'~/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/{pid}_features.pt')
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found {audio_path}")
        return

    print(f"\nLoading {pid}...")
    # Load raw
    audio = torch.load(audio_path, map_location='cpu')
    text = torch.load(text_path, map_location='cpu')
    
    print(f"Original Audio Shape: {audio.shape}, Dtype: {audio.dtype}")
    print(f"Original Text Shape:  {text.shape}, Dtype: {text.dtype}")
    
    # Simulate Dataset Processing
    # Slice first 8 steps
    audio_win = audio[0:8].float()
    text_win = text[0:8].float()
    
    print(f"\nWindow Audio Shape: {audio_win.shape}")
    print(f"Window Text Shape:  {text_win.shape}")
    
    # Check for 4D issue
    # If shape is [8, 1, 1024], we need squeeze
    if len(audio_win.shape) == 3 and audio_win.shape[1] == 1:
        print("Detected 3D tensor with singleton dim (common issue).")
    
    # Create Batch
    audio_batch = audio_win.unsqueeze(0) # [1, 8, 1024]
    text_batch = text_win.unsqueeze(0)   # [1, 8, 768]
    scalars = torch.randn(1, 3)
    
    print(f"\nBatch Input Shape: {audio_batch.shape}")
    
    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing Device: {device}")
    
    model = FusionV10Model(hidden_dim=256, num_layers=2).to(device)
    
    audio_batch = audio_batch.to(device)
    text_batch = text_batch.to(device)
    scalars = scalars.to(device)
    
    # 3. Forward Pass
    try:
        print("Running Forward Pass...")
        logits = model(audio_batch, text_batch, scalars)
        print("\nSUCCESS! Logits shape:", logits.shape)
    except Exception as e:
        print("\nFAILED during Forward Pass!")
        print(e)
        
        # Debugging suggestion
        if "4D instead" in str(e):
             print("\nDEBUG: The LSTM received a 4D tensor. This is likely because the input had an extra dimension.")
             print("Please check if the original features are [T, 1, D] instead of [T, D].")

if __name__ == "__main__":
    main()
