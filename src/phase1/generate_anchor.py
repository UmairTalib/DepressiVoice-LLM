#!/usr/bin/env python3
"""
Biometric Anchor Generation Script
Creates the Ellie reference embedding from confirmed high-quality interviewer audio
"""

import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from tqdm import tqdm
import os

def extract_interviewer_embedding(audio_file, embedding_model):
    """Extract embedding from interviewer audio"""
    try:
        # Load audio
        waveform, sr = sf.read(audio_file)
        
        # Handle stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Convert to tensor
        waveform_tensor = torch.from_numpy(waveform).float()
        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = embedding_model(waveform_tensor)
        
        if hasattr(embedding, 'detach'):
            return embedding.detach().cpu().numpy()
        return embedding
    
    except Exception as e:
        print(f"  ✗ Failed to process {audio_file.name}: {e}")
        return None

def generate_anchor(master_audio_dir, output_file, num_reference_sessions=10):
    """Generate biometric anchor from reference sessions"""
    
    print("\n=== BIOMETRIC ANCHOR GENERATION ===")
    
    # Load HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Load embedding model
    print("Loading speaker embedding model...")
    embedding_model = PretrainedSpeakerEmbedding(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token=hf_token
    )
    
    # Select reference sessions (use first N sessions for consistency)
    audio_files = sorted(Path(master_audio_dir).glob("*_AUDIO.wav"))[:num_reference_sessions]
    
    print(f"Processing {len(audio_files)} reference sessions...")
    
    # Extract embeddings
    embeddings = []
    for audio_file in tqdm(audio_files, desc="Extracting embeddings"):
        embedding = extract_interviewer_embedding(audio_file, embedding_model)
        if embedding is not None:
            embeddings.append(embedding.flatten())
    
    if len(embeddings) < 5:
        raise ValueError(f"Too few valid embeddings ({len(embeddings)}). Need at least 5.")
    
    # Compute centroid (average embedding)
    anchor_embedding = np.mean(embeddings, axis=0)
    
    # Save anchor
    anchor_data = {
        "embedding": anchor_embedding.tolist(),
        "num_reference_sessions": len(embeddings),
        "reference_sessions": [f.stem.replace("_AUDIO", "") for f in audio_files[:len(embeddings)]],
        "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "method": "centroid"
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(anchor_data, f, indent=2)
    
    print(f"\n=== ANCHOR GENERATION COMPLETE ===")
    print(f"Reference sessions used: {len(embeddings)}")
    print(f"Anchor saved to: {output_path}")
    print(f"Embedding dimension: {len(anchor_embedding)}")
    
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Ellie biometric anchor")
    parser.add_argument("--master_audio_dir", required=True, help="Directory containing master audio files")
    parser.add_argument("--output_file", required=True, help="Output anchor JSON file")
    parser.add_argument("--num_reference", type=int, default=10, help="Number of reference sessions")
    
    args = parser.parse_args()
    
    generate_anchor(args.master_audio_dir, args.output_file, args.num_reference)
