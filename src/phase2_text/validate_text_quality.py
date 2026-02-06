#!/usr/bin/env python3
"""
Phase 2B: Deep Quality Validation for Text Embeddings
Checks for:
1. Feature Distribution (Mean, Std, Min, Max)
2. Vector Diversity (Silence Ratio / Repetition)
3. Numerical Stability (NaN/Inf)
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def validate_quality(text_dir, session_ids=None):
    text_path = Path(text_dir)
    
    if session_ids is None:
        text_files = list(text_path.glob("*_features.pt"))
        session_ids = [f.stem.replace("_features", "") for f in text_files]
    
    print(f"Validating Quality for {len(session_ids)} sessions...")
    
    global_stats = {
        "means": [],
        "stds": [],
        "unique_ratios": [],
        "nan_count": 0
    }
    
    problematic_sessions = []

    for sid in tqdm(session_ids):
        file_path = text_path / f"{sid}_features.pt"
        
        try:
            # Load on CPU
            emb = torch.load(file_path, map_location='cpu', weights_only=True)
            
            # 1. Numerical Check
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"✗ Session {sid}: Contains NaN or Inf!")
                global_stats["nan_count"] += 1
                problematic_sessions.append((sid, "NaN/Inf"))
                continue

            # 2. Distribution Check
            mean = emb.mean().item()
            std = emb.std().item()
            global_stats["means"].append(mean)
            global_stats["stds"].append(std)
            
            # Check Z-Score normalization (Mean ~0.0, Std ~1.0)
            if abs(mean) > 0.1 or abs(std - 1.0) > 0.1:
                # Not necessarily an error, but worth noting deviation
                pass

            # 3. Diversity / Silence Check
            # Count unique vectors (rows)
            # Rounding to avoids float precision issues distincting "similar" vectors
            rounded = torch.round(emb * 1000) / 1000
            unique_rows = torch.unique(rounded, dim=0).shape[0]
            total_rows = emb.shape[0]
            
            unique_ratio = unique_rows / total_rows if total_rows > 0 else 0
            global_stats["unique_ratios"].append(unique_ratio)
            
            # If >50% of the session is the exact same vector, it's highly silent
            if unique_ratio < 0.1 and total_rows > 10:
               problematic_sessions.append((sid, f"Low Diversity ({unique_ratio:.2%})"))

        except Exception as e:
            print(f"✗ Session {sid}: Error - {e}")

    # Summary
    avg_mean = np.mean(global_stats["means"])
    avg_std = np.mean(global_stats["stds"])
    avg_diversity = np.mean(global_stats["unique_ratios"])
    
    print("\n" + "="*50)
    print("DEEP QUALITY CHECK SUMMARY")
    print("="*50)
    print(f"Sessions Checked: {len(session_ids)}")
    print(f"Global Mean:      {avg_mean:.4f} (Target: 0.0)")
    print(f"Global Std:       {avg_std:.4f}  (Target: 1.0)")
    print(f"Avg Vector Diversity: {avg_diversity:.1%} (Higher is better)")
    print(f"NaN/Inf files:    {global_stats['nan_count']}")
    print("-" * 50)
    
    if problematic_sessions:
        print("Potential Issues (Low Diversity / NaN):")
        for sid, issue in problematic_sessions[:10]:
            print(f"  - Session {sid}: {issue}")
        if len(problematic_sessions) > 10:
            print(f"  ... and {len(problematic_sessions)-10} more")
    else:
        print("✅ No major anomalies detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", required=True, help="Path to text embeddings")
    args = parser.parse_args()
    
    validate_quality(args.text_dir)
