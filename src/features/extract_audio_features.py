"""
Feature Extraction Module (Refactored for Clean Audio)

This module extracts WavLM embeddings (Layers 6-12 mean) from 
verified clean patient-only audio files.
"""

import os
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        # Fallback default config if not found
        return {
            'audio_features': {
                'window_size': 8.0,
                'overlap': 0.0,
                'model': 'microsoft/wavlm-base-plus',
                'batch_size': 4
            }
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_features(session_id=None, input_dir=None, output_dir=None, model_name="microsoft/wavlm-base-plus"):
    # 1. Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model & Processor
    print(f"Loading model: {model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    # 3. Determine Sessions to process
    if session_id:
        sessions = [session_id]
    else:
        # If no session_id, look in input_dir
        sessions = [s for s in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, s))]
        sessions.sort()

    os.makedirs(output_dir, exist_ok=True)

    target_sr = 16000
    window_size_sec = 8.0 # Standard for our pipeline
    stride_sec = 8.0 # No overlap for raw feature extraction (Phase 3 handles sequence)

    for sid in tqdm(sessions, desc="Extracting Features"):
        save_path = os.path.join(output_dir, f"{sid}_features.pt")
        if os.path.exists(save_path):
            # print(f"Skipping {sid}, already exists.")
            continue

        # Path logic for Phase 1 results
        # Pattern: input_dir/{sid}/patient_audio_{sid}.wav
        session_subdir = os.path.join(input_dir, str(sid))
        audio_file = os.path.join(session_subdir, f"patient_audio_{sid}.wav")
        
        # Fallback to patient_audio.wav if specific name not found
        if not os.path.exists(audio_file):
            audio_file = os.path.join(session_subdir, "patient_audio.wav")

        if not os.path.exists(audio_file):
            print(f"WARNING: Audio not found for session {sid} at {audio_file}")
            continue

        try:
            # Load clean audio
            waveform_np, _ = librosa.load(audio_file, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np)
            
            # Sliding Window
            window_samples = int(window_size_sec * target_sr)
            stride_samples = int(stride_sec * target_sr)
            
            # Padding if too short
            if len(waveform) < window_samples:
                padding = window_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Unfold: (Num_Windows, Window_Samples)
            windows = waveform.unfold(0, window_samples, stride_samples)
            
            features_list = []
            
            # Batch process windows
            batch_size = 4
            for i in range(0, len(windows), batch_size):
                batch_audio = windows[i:i+batch_size].to(device)
                
                # Normalize
                # Processor expects list of arrays or a single array-like
                inputs_obj = processor(list(batch_audio.cpu().numpy()), sampling_rate=target_sr, return_tensors="pt", padding=True)
                inputs = inputs_obj.input_values.to(device)
                
                # Ensure shape is (Batch, Time)
                if inputs.ndim == 3 and inputs.shape[0] == 1:
                    inputs = inputs.squeeze(0)
                elif inputs.ndim == 4:
                    # Handle unexpected nesting [1, 1, B, T]
                    inputs = inputs.squeeze(0).squeeze(0)

                with torch.no_grad():
                    outputs = model(inputs)
                    
                    # Extract Layers 6-12 (indices 6:13)
                    # hidden_states: (Layer, Batch, SeqLen, Hidden)
                    all_hidden = outputs.hidden_states
                    selected_layers = torch.stack(all_hidden[6:13], dim=0)
                    
                    # Mean across layers: (Batch, SeqLen, 768)
                    layer_mean = torch.mean(selected_layers, dim=0)
                    
                    features_list.append(layer_mean.cpu())

            if features_list:
                # Cat over windows: (Total_Windows, SeqLen, 768)
                all_features = torch.cat(features_list, dim=0)
                
                # Save as .pt
                save_path = os.path.join(output_dir, f"{sid}_features.pt")
                torch.save(all_features, save_path)
                # print(f"Saved {sid}: {all_features.shape}")

        except Exception as e:
            print(f"Error session {sid}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, help="Specific session to process")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Phase1_results")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .pt files")
    parser.add_argument("--model", type=str, default="microsoft/wavlm-base-plus")
    args = parser.parse_args()
    
    extract_features(
        session_id=args.session_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model
    )
