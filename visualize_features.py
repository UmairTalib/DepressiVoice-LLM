import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_embeddings(pt_file):
    print(f"Loading {pt_file}...")
    # Load features: Shape [N_Windows, Time, Digits]
    # We want to visualize ONE window to see the speech patterns
    features = torch.load(pt_file, map_location="cpu")
    
    # Pick the first window (8 seconds)
    # Shape: [399, 768] (Time x Features)
    # We transpose to [768, 399] so Time is on X-axis
    window_data = features[0].T.numpy()
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(window_data, cmap="viridis", cbar=True, vmin=-3, vmax=3)
    
    plt.title(f"WavLM Feature Embeddings (Single 8s Window)\nShape: {features[0].shape}")
    plt.xlabel("Time Steps (approx 20ms each)")
    plt.ylabel("Feature Dimensions (0-768)")
    
    output_file = "embedding_visualization.png"
    plt.savefig(output_file)
    print(f"✅ Visualization saved to {output_file}")
    print("If you see vertical stripes or patterns, the embedding is capturing speech!")

if __name__ == "__main__":
    visualize_embeddings("302_features.pt")
