"""
Phase 2 Audio Feature Validation Utilities

Provides validation functions for WavLM-Large feature extraction outputs.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def validate_features(feature_path, save_plot=True):
    """
    Validate extracted audio features for quality and correctness.
    
    Args:
        feature_path: Path to .pt feature file
        save_plot: If True, save diagnostic visualization
        
    Returns:
        dict: Validation statistics
    """
    print(f"Loading: {feature_path}")
    features = torch.load(feature_path, map_location='cpu', weights_only=False).float()
    
    print(f"\n{'='*60}")
    print("FEATURE VALIDATION")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n1. Basic Statistics:")
    print(f"   Shape: {features.shape}")
    print(f"   Dtype: {features.dtype}")
    
    # Value range
    print(f"\n2. Value Range Analysis:")
    print(f"   Global Max: {features.max().item():.4f}")
    print(f"   Global Min: {features.min().item():.4f}")
    print(f"   Global Mean: {features.mean().item():.4f}")
    print(f"   Global Std: {features.std().item():.4f}")
    
    # NaN/Inf check
    has_nan = torch.isnan(features).any()
    has_inf = torch.isinf(features).any()
    print(f"\n3. Data Quality:")
    print(f"   Contains NaN: {has_nan}")
    print(f"   Contains Inf: {has_inf}")
    
    # Per-dimension analysis
    print(f"\n4. Dimension-wise Statistics:")
    features_flat = features.reshape(-1, features.shape[-1])
    
    dim_max = features_flat.max(dim=0).values
    dim_min = features_flat.min(dim=0).values
    dim_mean = features_flat.mean(dim=0)
    dim_std = features_flat.std(dim=0)
    
    # Find potential issues
    rogue_threshold = 100.0
    rogue_dims = torch.where(dim_max > rogue_threshold)[0]
    
    print(f"   Potential Issues (max > {rogue_threshold}): {len(rogue_dims)}")
    if len(rogue_dims) > 0:
        print(f"   Top 5 Dimension Indices:")
        top_sorted = torch.argsort(dim_max, descending=True)[:5]
        for idx in top_sorted:
            print(f"      Dim {idx}: max={dim_max[idx]:.2f}, mean={dim_mean[idx]:.2f}")
    
    # Generate visualization if requested
    if save_plot:
        output_path = str(feature_path).replace('.pt', '_validation.png')
        _generate_diagnostic_plot(features, dim_max, dim_std, output_path)
        print(f"\n5. Visualization saved: {output_path}")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf),
        'max_value': features.max().item(),
        'min_value': features.min().item(),
        'mean_value': features.mean().item(),
        'std_value': features.std().item(),
        'num_rogue_dims': len(rogue_dims),
        'shape': features.shape
    }

def _generate_diagnostic_plot(features, dim_max, dim_std, output_path):
    """Generate 4-panel diagnostic visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Max values per dimension
    axes[0, 0].plot(dim_max.numpy())
    axes[0, 0].set_title('Max Value per Dimension')
    axes[0, 0].set_xlabel('Dimension Index')
    axes[0, 0].set_ylabel('Max Value')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', label='Threshold (100)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature heatmap (clipped)
    sample = features[0, :, :].numpy()
    im1 = axes[0, 1].imshow(sample.T, aspect='auto', cmap='viridis', vmin=-50, vmax=50)
    axes[0, 1].set_title('Feature Heatmap (Window 0, clipped ±50)')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Feature Dimension')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Full-scale heatmap
    im2 = axes[1, 0].imshow(sample.T, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Feature Heatmap (Window 0, full scale)')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Feature Dimension')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Std dev per dimension
    axes[1, 1].plot(dim_std.numpy())
    axes[1, 1].set_title('Standard Deviation per Dimension')
    axes[1, 1].set_xlabel('Dimension Index')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def batch_validate(input_dir, num_samples=10):
    """
    Validate multiple feature files from a directory.
    
    Args:
        input_dir: Directory containing .pt feature files
        num_samples: Number of random samples to validate
        
    Returns:
        dict: Summary statistics across all samples
    """
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*_features.pt"))
    
    if len(files) == 0:
        print(f"No feature files found in {input_dir}")
        return None
    
    # Sample random files
    import random
    sample_files = random.sample(files, min(num_samples, len(files)))
    
    print(f"Validating {len(sample_files)} samples from {len(files)} total files...")
    
    results = []
    for file_path in sample_files:
        result = validate_features(file_path, save_plot=False)
        results.append(result)
    
    # Aggregate statistics
    summary = {
        'total_files': len(files),
        'samples_validated': len(sample_files),
        'all_valid': all(not r['has_nan'] and not r['has_inf'] for r in results),
        'max_value_range': (min(r['max_value'] for r in results), 
                           max(r['max_value'] for r in results)),
        'mean_value_range': (min(r['mean_value'] for r in results),
                             max(r['mean_value'] for r in results)),
    }
    
    print(f"\n{'='*60}")
    print("BATCH VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {summary['total_files']}")
    print(f"Samples validated: {summary['samples_validated']}")
    print(f"All valid: {summary['all_valid']}")
    print(f"Max value range: {summary['max_value_range'][0]:.2f} to {summary['max_value_range'][1]:.2f}")
    print(f"{'='*60}\n")
    
    return summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate.py <feature_file.pt>")
        print("   Or: python validate.py <directory> --batch")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if '--batch' in sys.argv:
        batch_validate(path)
    else:
        validate_features(path)
