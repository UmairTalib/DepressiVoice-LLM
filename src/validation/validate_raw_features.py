#!/usr/bin/env python3
"""
Validate extracted features and identify anisotropy
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def validate_features(feature_path):
    print(f"Loading: {feature_path}")
    features = torch.load(feature_path, map_location='cpu').float()
    
    print(f"\n{'='*60}")
    print("FEATURE VALIDATION")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n1. Basic Statistics:")
    print(f"   Shape: {features.shape}")
    print(f"   Dtype: {features.dtype}")
    print(f"   Device: {features.device}")
    
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
    # Collapse time dimensions: [Windows, Frames, Dims] -> [Dims]
    features_flat = features.reshape(-1, features.shape[-1])
    
    dim_max = features_flat.max(dim=0).values
    dim_min = features_flat.min(dim=0).values
    dim_mean = features_flat.mean(dim=0)
    dim_std = features_flat.std(dim=0)
    
    # Find rogue dimensions
    rogue_threshold = 100.0
    rogue_dims = torch.where(dim_max > rogue_threshold)[0]
    
    print(f"   Rogue Dimensions (max > {rogue_threshold}): {len(rogue_dims)}")
    if len(rogue_dims) > 0:
        print(f"   Top 5 Rogue Dimension Indices:")
        top_rogue_sorted = torch.argsort(dim_max, descending=True)[:5]
        for idx in top_rogue_sorted:
            print(f"      Dim {idx}: max={dim_max[idx]:.2f}, min={dim_min[idx]:.2f}, mean={dim_mean[idx]:.2f}")
    
    # Visualization
    print(f"\n5. Generating Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Max values per dimension
    axes[0, 0].plot(dim_max.numpy())
    axes[0, 0].set_title('Max Value per Dimension')
    axes[0, 0].set_xlabel('Dimension Index')
    axes[0, 0].set_ylabel('Max Value')
    axes[0, 0].axhline(y=rogue_threshold, color='r', linestyle='--', label=f'Threshold ({rogue_threshold})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature heatmap (sample)
    sample_features = features[0, :, :].numpy()  # First window
    im1 = axes[0, 1].imshow(sample_features.T, aspect='auto', cmap='viridis', vmin=-50, vmax=50)
    axes[0, 1].set_title('Feature Heatmap (Window 0, clipped to ±50)')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Feature Dimension')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Unclamped heatmap (showing anisotropy)
    im2 = axes[1, 0].imshow(sample_features.T, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Feature Heatmap (Window 0, full scale)')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Feature Dimension')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Standard deviation per dimension
    axes[1, 1].plot(dim_std.numpy())
    axes[1, 1].set_title('Standard Deviation per Dimension')
    axes[1, 1].set_xlabel('Dimension Index')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = feature_path.replace('.pt', '_validation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization: {output_path}")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'max_value': features.max().item(),
        'num_rogue_dims': len(rogue_dims),
        'shape': features.shape
    }

if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else "632"
    feature_file = f"/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_raw/{session}_features.pt"
    validate_features(feature_file)
