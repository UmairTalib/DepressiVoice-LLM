#!/usr/bin/env python3
"""
Batch Normalize WavLM Features (Phase 2.5)
Applies Layer Normalization to all extracted features to fix anisotropy
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import os

def batch_normalize(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all feature files
    files = sorted(input_path.glob("*_features.pt"))
    print(f"Found {len(files)} files to normalize.")
    
    if len(files) == 0:
        print(f"ERROR: No files found in {input_dir}")
        return
    
    stats = {
        'original_max': [],
        'original_min': [],
        'normalized_max': [],
        'normalized_min': []
    }
    
    for i, file_path in enumerate(tqdm(files, desc="Normalizing")):
        try:
            # Load
            features = torch.load(file_path, map_location='cpu').float()
            
            # Record original stats
            orig_max = features.max().item()
            orig_min = features.min().item()
            stats['original_max'].append(orig_max)
            stats['original_min'].append(orig_min)
            
            # Normalize (across last dimension: 1024)
            normalized = F.layer_norm(features, (features.shape[-1],))
            
            # Record normalized stats
            norm_max = normalized.max().item()
            norm_min = normalized.min().item()
            stats['normalized_max'].append(norm_max)
            stats['normalized_min'].append(norm_min)
            
            # Convert to FP16 for storage
            normalized_fp16 = normalized.to(torch.float16)
            
            # Save
            output_file = output_path / file_path.name
            torch.save(normalized_fp16, output_file)
            
            # Verification log for first file
            if i == 0:
                print(f"\n[Sample: {file_path.name}]")
                print(f"  Original: max={orig_max:.2f}, min={orig_min:.2f}")
                print(f"  Normalized: max={norm_max:.2f}, min={norm_min:.2f}")
                print(f"  Shape: {features.shape}")
                print("-" * 60)
        
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("NORMALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(files)}")
    print(f"\nOriginal Range:")
    print(f"  Max: {max(stats['original_max']):.2f} (worst case)")
    print(f"  Min: {min(stats['original_min']):.2f}")
    print(f"\nNormalized Range:")
    print(f"  Max: {max(stats['normalized_max']):.2f}")
    print(f"  Min: {min(stats['normalized_min']):.2f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    INPUT_DIR = "/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_raw"
    OUTPUT_DIR = "/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized"
    
    batch_normalize(INPUT_DIR, OUTPUT_DIR)
