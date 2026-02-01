import argparse
import os
import torch
import numpy as np
from pyannote.audio import Inference
import torchaudio

def generate_anchor(audio_path, output_path, hf_token):
    """
    Generates a speaker embedding anchor from a reference audio file.
    """
    print(f"INFO: Loading embedding model...")
    # Using the standard embedding model from pyannote 3.1
    # This matches what is likely used under the hood or is compatible
    model = Inference(
        "pyannote/wespeaker-voxceleb-resnet34-LM", 
        use_auth_token=hf_token
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"INFO: Model loaded on {device}")

    print(f"INFO: Processing audio {audio_path}...")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    # the inference model handles loading and cropping internally usually, 
    # but providing the path is the standard way for pyannote.audio Inference
    embedding = model(audio_path)
    
    # embedding is a (num_windows, dimension) array or (dimension,) depending on usage
    # If the file is long, Inference might return a SlidingWindowFeature
    # Let's handle generic case:
    
    # Actually, Inference(path) typically returns a single embedding for the whole file 
    # if window="whole" is implied or configured, but standard usage might imply sliding window.
    # Let's explicitly average if it returns multiple.
    
    print(f"INFO: Embedding shape: {embedding.shape}")
    
    if len(embedding.shape) > 1:
         # Average over time if multiple windows
        anchor_embedding = np.mean(embedding, axis=0)
    else:
        anchor_embedding = embedding

    # Normalize just in case, though cosine similarity handles magnitude
    norm = np.linalg.norm(anchor_embedding)
    if norm > 0:
        anchor_embedding = anchor_embedding / norm

    print(f"INFO: Saving anchor to {output_path}")
    np.save(output_path, anchor_embedding)
    print("SUCCESS: Anchor generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Speaker Anchor Embedding")
    parser.add_argument("--audio", required=True, help="Path to reference audio file (Ellie)")
    parser.add_argument("--output", default="data/ellie_anchor.npy", help="Path to save .npy anchor")
    parser.add_argument("--token", help="Hugging Face Token", default=os.environ.get("HF_TOKEN"))
    
    args = parser.parse_args()
    
    if not args.token:
        print("ERROR: HF_TOKEN not found. Set it in env or pass --token")
        exit(1)
        
    generate_anchor(args.audio, args.output, args.token)
