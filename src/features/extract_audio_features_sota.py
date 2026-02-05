"""
SOTA Feature Extraction Module (2026 Standards)

Upgrades:
1. Backbone: microsoft/wavlm-large (315M parameters)
2. Hybrid Pooling: Softmax-normalized Weighted Layer Sum (Layers 6-12)
3. Precision: Float16 for storage efficiency
4. Force-Write: Re-calculates all sessions to ensure integrity
"""

import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

def extract_features(session_id=None, input_dir=None, output_dir=None, model_name="microsoft/wavlm-large"):
    # 1. Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model & Processor
    print(f"Loading SOTA model: {model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    # 3. Define Layer Weights (Softmax-normalized)
    # Rationale: Higher layers (10-12) capture more paralinguistic/prosodic nuances
    # for psychiatric biomarkers than middle layers (6-7).
    # We initialize with a bias towards higher layers.
    layer_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    normalized_weights = torch.softmax(layer_weights, dim=0)
    print(f"Using Weighted Layer Sum. Normalized Weights (L6-L12): {normalized_weights.cpu().numpy()}")

    # 4. Determine Sessions
    if session_id:
        sessions = [session_id]
    else:
        sessions = [s for s in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, s))]
        sessions.sort()

    os.makedirs(output_dir, exist_ok=True)

    target_sr = 16000
    window_size_sec = 8.0 
    stride_sec = 8.0 

    for sid in tqdm(sessions, desc="Extracting Features"):
        session_subdir = os.path.join(input_dir, str(sid))
        audio_file = os.path.join(session_subdir, "patient_audio.wav")

        if not os.path.exists(audio_file):
            continue

        try:
            # Load audio
            waveform_np, _ = librosa.load(audio_file, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np)
            
            # Sliding Window
            window_samples = int(window_size_sec * target_sr)
            stride_samples = int(stride_sec * target_sr)
            
            if len(waveform) < window_samples:
                padding = window_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            windows = waveform.unfold(0, window_samples, stride_samples)
            
            features_list = []
            
            # Batch process windows
            batch_size = 2 # Reduced for WavLM-Large memory safety
            for i in range(0, len(windows), batch_size):
                batch_audio = windows[i:i+batch_size].to(device)
                
                inputs_obj = processor(list(batch_audio.cpu().numpy()), sampling_rate=target_sr, return_tensors="pt", padding=True)
                inputs = inputs_obj.input_values.to(device)
                
                # Squeeze nested dims if present
                if inputs.ndim == 3 and inputs.shape[0] == 1:
                    inputs = inputs.squeeze(0)

                with torch.no_grad():
                    outputs = model(inputs)
                    
                    # Extract Layers 6-12 (indices 6 to 12 inclusive -> hidden_states are 0-indexed)
                    # WavLM-Large has 25 hidden states (0=input bits, 1-24=layers)
                    # Layer 6 is index 6, Layer 12 is index 12.
                    all_hidden = outputs.hidden_states
                    
                    # Stack selected layers: (7, Batch, SeqLen, 1024) - Large is 1024 dim
                    selected_layers = torch.stack(all_hidden[6:13], dim=0)
                    
                    # Apply Weighted Sum
                    # (7, 1, 1, 1) * (7, Batch, SeqLen, 1024) -> Sum over 0-dim
                    weighted_sum = torch.sum(selected_layers * normalized_weights.view(-1, 1, 1, 1), dim=0)
                    
                    # Convert to Float16 for storage efficiency
                    features_list.append(weighted_sum.to(torch.float16).cpu())

            if features_list:
                all_features = torch.cat(features_list, dim=0)
                save_path = os.path.join(output_dir, f"{sid}_features.pt")
                torch.save(all_features, save_path)

        except Exception as e:
            print(f"Error session {sid}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, help="Specific session to process")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Phase1_results")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .pt files")
    parser.add_argument("--model", type=str, default="microsoft/wavlm-large")
    args = parser.parse_args()
    
    extract_features(
        session_id=args.session_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model
    )
