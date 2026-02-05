import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def check_normalized_patterns(pt_path, output_path):
    # 1. Load data
    features = torch.load(pt_path, map_location='cpu', weights_only=True).float()
    
    # 2. Before Normalization (Original)
    orig_window = features[0].T.numpy()
    
    # 3. Apply Layer Normalization
    # We normalize across the feature dimension (last dim)
    ln = nn.LayerNorm(1024)
    with torch.no_grad():
        normalized_features = ln(features)
    
    norm_window = normalized_features[0].T.numpy()
    
    # 4. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    
    # Original
    im1 = ax1.imshow(orig_window, aspect='auto', cmap='viridis')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f"ORIGINAL (Session {os.path.basename(pt_path)})\nDominated by Rogue Dimensions (Note colorbar scale)")
    
    # Normalized
    im2 = ax2.imshow(norm_window, aspect='auto', cmap='viridis', vmin=-3, vmax=3)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f"LAYER-NORMALIZED\nOutliers suppressed. Vertical Speech Patterns (Texture) Revealed!")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    check_normalized_patterns("/Users/umairtalib/.gemini/antigravity/brain/5beebc64-a2b7-4ee3-8674-cb027cbea6ca/300_features_sota.pt", 
                             "/Users/umairtalib/.gemini/antigravity/brain/5beebc64-a2b7-4ee3-8674-cb027cbea6ca/rogue_dim_fix_verification.png")
