"""
Batch Normalize WavLM Features (Phase 2.5)
Resolves systemic anisotropy (Rogue Dimensions) by applying Layer Normalization.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import os

def clean_and_save_features(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_path.glob("*.pt")))
    print(f"Found {len(files)} files to process.")

    for i, file_path in enumerate(tqdm(files)):
        # 1. Load
        # WavLM-Large SOTA features are saved as float16
        features = torch.load(file_path, map_location='cpu', weights_only=True).float()
        
        # 2. Normalize (Fixing the Anisotropy)
        # We normalize over the last dimension (1024)
        # Result: Mean ~0, Std ~1. Outliers squashed.
        normalized_features = F.layer_norm(features, (1024,))
        
        # 3. Save as float16 to maintain efficiency
        save_name = file_path.stem + "_norm.pt"
        torch.save(normalized_features.to(torch.float16), output_path / save_name)
        
        # Verification log for the first file
        if i == 0:
            print(f"\n[Verification] {file_path.name}")
            print(f"Original Max: {features.max().item():.2f}")
            print(f"Cleaned Max:  {normalized_features.max().item():.2f}")
            print(f"Original Min: {features.min().item():.2f}")
            print(f"Cleaned Min:  {normalized_features.min().item():.2f}")
            print("--------------------------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw embeddings")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned embeddings")
    args = parser.parse_args()
    
    clean_and_save_features(args.input, args.output)
