"""
Regenerate Audio Local
Slice raw audio files using clean metadata timestamps to remove interviewer features.
"""

import os
import json
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import argparse
from pathlib import Path

def regenerate_audio(raw_audio_dir, metadata_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Map session IDs from metadata
    # Metadata structure: data/processed/phase1/session_outputs/{ID}/cleaned_patient_text_v2.json
    metadata_path = Path(metadata_dir)
    sessions = [d.name for d in metadata_path.iterdir() if d.is_dir() and d.name.isdigit()]
    sessions.sort()
    
    print(f"Found {len(sessions)} sessions with metadata.")
    
    success_count = 0
    missing_raw = []
    
    for sid in tqdm(sessions, desc="Regenerating Audio"):
        # Paths
        json_path = metadata_path / sid / "cleaned_patient_text_v2.json"
        raw_wav_path = Path(raw_audio_dir) / f"{sid}_AUDIO.wav"
        out_wav_path = Path(output_dir) / f"patient_audio_{sid}.wav"
        
        # Check raw existence
        if not raw_wav_path.exists():
            missing_raw.append(sid)
            continue
            
        try:
            # Load Metadata
            with open(json_path, 'r') as f:
                meta = json.load(f)
                
            segments = meta.get('segments', [])
            if not segments:
                print(f"Warning: No segments for {sid}")
                continue
                
            # Load Raw Audio (scipy is faster for simple slicing)
            # Check file size first? No, scipy uses mmap if needed or reads header.
            # Reading entire file into memory might be heavy (50MB+), but manageable for 8GB+ RAM.
            sample_rate, data = wavfile.read(raw_wav_path)
            
            # If stereo, convert to mono? Use channel 0 (usually valid for DAIC-WOZ)
            if len(data.shape) > 1:
                data = data.mean(axis=1).astype(data.dtype)
                
            # Slice and Concatenate
            clean_audio_parts = []
            
            for seg in segments:
                start_sec = seg['start']
                end_sec = seg['end']
                
                # Convert to samples
                start_idx = int(start_sec * sample_rate)
                end_idx = int(end_sec * sample_rate)
                
                # Clamp
                start_idx = max(0, start_idx)
                end_idx = min(len(data), end_idx)
                
                if start_idx < end_idx:
                    clean_audio_parts.append(data[start_idx:end_idx])
            
            if clean_audio_parts:
                final_audio = np.concatenate(clean_audio_parts)
                wavfile.write(out_wav_path, sample_rate, final_audio)
                success_count += 1
            else:
                print(f"Warning: Resulting audio empty for {sid}")
                
        except Exception as e:
            print(f"Error processing {sid}: {e}")
            
    print("\n" + "="*40)
    print(f"Regeneration Complete.")
    print(f"Successfully Created: {success_count}/{len(sessions)}")
    print(f"Missing Raw Files: {len(missing_raw)}")
    if missing_raw:
        print(f"Missing IDs: {missing_raw[:10]}...")
    print(f"Output Directory: {output_dir}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw _AUDIO.wav files")
    parser.add_argument("--meta_dir", type=str, default="data/processed/phase1/session_outputs", help="Path containing session folders with json")
    parser.add_argument("--out_dir", type=str, default="data/repaired_audio", help="Output for clean wavs")
    args = parser.parse_args()
    
    regenerate_audio(args.raw_dir, args.meta_dir, args.out_dir)
