import torch
import matplotlib.pyplot as plt
import os

def visualize_sota(pt_dir, output_dir="data/04_results/validation/plots_sota"):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])[:3]
    
    for f in files:
        path = os.path.join(pt_dir, f)
        features = torch.load(path, map_location='cpu', weights_only=True)
        # pick first window, transpose for Time on X-axis
        # [N, 399, 1024] -> [1024, 399]
        window_data = features[0].T.float().numpy()
        
        plt.figure(figsize=(15, 10))
        plt.imshow(window_data, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Activation Intensity')
        plt.title(f"SOTA WavLM-Large Features: {f}\nShape: {features[0].shape}")
        plt.xlabel("Time Steps (~20ms)")
        plt.ylabel("Dimensions (0-1023)")
        
        save_path = os.path.join(output_dir, f"{f.replace('.pt', '.png')}")
        plt.savefig(save_path)
        plt.close()
        print(f"Generated heatmap: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    visualize_sota(args.dir)
