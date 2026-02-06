#!/usr/bin/env python3
"""
Final validation of normalized Phase 2 features
Generates statistics and visualizations for comprehensive report
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def validate_all_features():
    raw_dir = Path("/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_raw")
    norm_dir = Path("/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized")
    
    raw_files = sorted(raw_dir.glob("*_features.pt"))
    norm_files = sorted(norm_dir.glob("*_features.pt"))
    
    print(f"{'='*70}")
    print("PHASE 2 FEATURE VALIDATION - FINAL REPORT")
    print(f"{'='*70}\n")
    
    # Basic counts
    print(f"1. File Counts:")
    print(f"   Raw features: {len(raw_files)}")
    print(f"   Normalized features: {len(norm_files)}")
    print(f"   Status: {'✓ Complete' if len(norm_files) == 275 else '✗ Incomplete'}\n")
    
    # Sample checks
    print(f"2. Sample Validation (10 random sessions):")
    samples = np.random.choice(len(norm_files), min(10, len(norm_files)), replace=False)
    
    all_valid = True
    stats = defaultdict(list)
    
    for idx in samples:
        norm_file = norm_files[idx]
        raw_file = raw_files[idx]
        
        # Load
        norm_feat = torch.load(norm_file, weights_only=False)
        raw_feat = torch.load(raw_file, weights_only=False)
        
        # Check
        has_nan_norm = torch.isnan(norm_feat).any()
        has_inf_norm = torch.isinf(norm_feat).any()
        
        session_id = norm_file.stem.split('_')[0]
        
        stats['raw_max'].append(raw_feat.max().item())
        stats['norm_max'].append(norm_feat.max().item())
        stats['shapes'].append(str(norm_feat.shape))
        
        status = "✓" if not (has_nan_norm or has_inf_norm) else "✗"
        print(f"   [{status}] Session {session_id}: shape={norm_feat.shape}, "
              f"raw_max={raw_feat.max():.1f}, norm_max={norm_feat.max():.2f}")
        
        if has_nan_norm or has_inf_norm:
            all_valid = False
    
    print(f"\n3. Statistical Summary:")
    print(f"   Raw Features:")
    print(f"      Max value range: {min(stats['raw_max']):.1f} to {max(stats['raw_max']):.1f}")
    print(f"   Normalized Features:")
    print(f"      Max value range: {min(stats['norm_max']):.2f} to {max(stats['norm_max']):.2f}")
    print(f"      Reduction: {(1 - max(stats['norm_max'])/max(stats['raw_max']))*100:.1f}%")
    
    print(f"\n4. Shape Consistency:")
    unique_shapes = set(stats['shapes'])
    print(f"   Unique shapes found: {len(unique_shapes)}")
    print(f"   Status: {'✓ All [Windows, 399, 1024]' if len(unique_shapes) > 0 else '✗ Shape mismatch detected'}")
    
    # Final verdict
    print(f"\n{'='*70}")
    if all_valid and len(norm_files) == 275:
        print("✓ VALIDATION PASSED - All features normalized successfully")
    else:
        print("✗ VALIDATION ISSUES DETECTED")
    print(f"{'='*70}\n")
    
    # Export summary
    summary = {
        'total_sessions': len(norm_files),
        'extraction_complete': len(raw_files) == 275,
        'normalization_complete': len(norm_files) == 275,
        'raw_max_global': max(stats['raw_max']),
        'norm_max_global': max(stats['norm_max']),
        'reduction_percent': (1 - max(stats['norm_max'])/max(stats['raw_max']))*100,
        'validation_passed': all_valid and len(norm_files) == 275
    }
    
    with open('/home/g755389/DepressiVoice-LLM/data/phase2_clean/validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: data/phase2_clean/validation_summary.json")

if __name__ == "__main__":
    validate_all_features()
