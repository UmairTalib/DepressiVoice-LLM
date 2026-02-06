#!/usr/bin/env python3
"""
Apply Layer Normalization to fix WavLM anisotropy
Normalizes across the 1024-dimensional feature space
"""

import torch
import torch.nn.functional as F
import sys

def normalize_features(input_path, output_path):
    print(f"Loading: {input_path}")
    features = torch.load(input_path, map_location='cpu').float()
    
    print(f"Original shape: {features.shape}")
    print(f"Original max: {features.max().item():.2f}, min: {features.min().item():.2f}")
    
    # Apply Layer Normalization across the feature dimension (1024)
    # Shape: [Windows, Frames, 1024] -> normalize over last dim
    normalized = F.layer_norm(features, (features.shape[-1],))
    
    print(f"Normalized max: {normalized.max().item():.2f}, min: {normalized.min().item():.2f}")
    print(f"Normalized mean: {normalized.mean().item():.4f}, std: {normalized.std().item():.4f}")
    
    # Convert back to FP16 for storage
    normalized_fp16 = normalized.to(torch.float16)
    
    # Save
    torch.save(normalized_fp16, output_path)
    
    print(f"Saved to: {output_path}")
    print(f"✓ Normalization complete!")
    
    return normalized_fp16

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "632"
    
    input_file = f"/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_raw/{session}_features.pt"
    output_file = f"/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/{session}_features.pt"
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    normalize_features(input_file, output_file)
