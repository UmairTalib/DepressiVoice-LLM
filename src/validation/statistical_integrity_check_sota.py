"""
SOTA Statistical Integrity Check for WavLM-Large Features
Updated for 1024-dim, Float16 embeddings.
"""

import os
import torch
import sys
from pathlib import Path

def check_file_integrity(filepath):
    result = {
        'filename': os.path.basename(filepath),
        'valid': True,
        'errors': [],
        'shape': None,
        'stats': {}
    }
    
    try:
        # Load tensor
        data = torch.load(filepath, map_location='cpu', weights_only=True)
        result['shape'] = tuple(data.shape)
        
        # Check 1: Shape Consistency - Last dimension must be 1024 for WavLM-Large
        if len(data.shape) < 3 or data.shape[2] != 1024:
            result['valid'] = False
            result['errors'].append(f"Invalid shape: {data.shape}, expected (N, 399, 1024)")
        
        # Check 2: Shape Consistency - Middle dimension must be 399
        if len(data.shape) >= 2 and data.shape[1] != 399:
            result['valid'] = False
            result['errors'].append(f"Invalid time dimension: {data.shape[1]}, expected 399")
        
        # Check 3: NaN/Inf Check
        has_nan = torch.isnan(data).any().item()
        has_inf = torch.isinf(data).any().item()
        
        if has_nan: result['valid'] = False; result['errors'].append("Contains NaN")
        if has_inf: result['valid'] = False; result['errors'].append("Contains Inf")
        
        # Check 4: Precision
        if data.dtype != torch.float16:
             result['valid'] = False; result['errors'].append(f"Invalid dtype: {data.dtype}, expected float16")

        # Stats (convert to float32 for stable calculation)
        data_f32 = data.float()
        std = data_f32.std().item()
        result['stats']['std'] = std
        result['stats']['mean'] = data_f32.mean().item()
        
        if std == 0.0:
            result['valid'] = False
            result['errors'].append("Dead signal: std = 0.0")
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Load error: {str(e)}")
    
    return result

def run_check(features_dir):
    pt_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.pt')])
    total = len(pt_files)
    print(f"Checking {total} SOTA files in {features_dir}...")
    
    results = []
    corrupted = []
    
    for idx, f in enumerate(pt_files, 1):
        res = check_file_integrity(f)
        results.append(res)
        if not res['valid']: corrupted.append(res)
        if idx % 50 == 0: print(f"  Processed {idx}/{total}...")

    print("\n" + "="*40)
    print("SOTA INTEGRITY SUMMARY")
    print(f"Total: {total} | Valid: {total - len(corrupted)} | Corrupted: {len(corrupted)}")
    print("="*40)
    
    if corrupted:
        for c in corrupted:
            print(f"❌ {c['filename']}: {', '.join(c['errors'])}")
    else:
        print("✅ 100% Integrity Pass Rate.")
        
    if results:
        valid = [r for r in results if r['valid']]
        stds = [r['stats']['std'] for r in valid]
        print(f"Avg Std: {sum(stds)/len(stds):.4f} (Min: {min(stds):.4f}, Max: {max(stds):.4f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    run_check(args.dir)
