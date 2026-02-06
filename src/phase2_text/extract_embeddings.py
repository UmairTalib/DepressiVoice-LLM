#!/usr/bin/env python3
"""
Phase 2B: Mental-RoBERTa Text Embedding Extraction
Extracts 768-dim embeddings with 8-second windowing to match audio features.

Critical: Window count MUST match audio for multimodal fusion in Phase 3.
"""

import argparse
import json
import torch
import librosa
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def extract_text_embeddings(session_id, input_dir, output_dir, model, verbose=False, skip_filtering=False):
    """
    Extract MentalRoBERTa embeddings with 8-second windowing.
    
    Args:
        session_id: Session ID (e.g., "300")
        input_dir: Path to phase1_rerun_outputs
        output_dir: Path to save embeddings
        model: Loaded SentenceTransformer model
        verbose: Print debug info
        skip_filtering: If True, use all transcript words without RTTM/Role checks
    
    Returns:
        tuple: (embeddings_tensor, num_windows, num_words)
    """
    # Try both "session_{id}" and "{id}" directory patterns
    session_dir = Path(input_dir) / f"session_{session_id}"
    if not session_dir.exists():
        session_dir = Path(input_dir) / f"{session_id}"
        
    # 1. Load audio to determine session duration and num_windows
    audio_file = session_dir / "patient_audio.wav"
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found in {session_dir}")
    
    audio, sr = librosa.load(str(audio_file), sr=16000)
    duration_sec = len(audio) / sr
    num_windows = int(duration_sec / 8.0)
    
    if verbose:
        print(f"Session {session_id}: Duration={duration_sec:.2f}s, Windows={num_windows}")
    
    patient_intervals = []
    
    # 2. Logic for filtering (skip if skip_filtering=True)
    if not skip_filtering:
        # Load Roles (Optional Fallback)
        roles_file = session_dir / "roles.json"
        patient_speaker_id = None
        if roles_file.exists():
            with open(roles_file) as f:
                roles_data = json.load(f)
            
            for spk, role in roles_data.get("roles", {}).items():
                if role.upper() == "PATIENT":
                    patient_speaker_id = spk
                    break
            
            if patient_speaker_id and verbose:
                print(f"  Patient ID: {patient_speaker_id}")
        
        # Load RTTM (Optional Fallback)
        if patient_speaker_id:
            rttm_file = session_dir / f"{session_id}.rttm"
            if not rttm_file.exists():
                rttm_files = list(session_dir.glob("*.rttm"))
                if rttm_files:
                    rttm_file = rttm_files[0]
                else:
                    rttm_file = None
            
            if rttm_file:
                with open(rttm_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[7] == patient_speaker_id:
                            start = float(parts[3])
                            duration = float(parts[4])
                            patient_intervals.append((start, start + duration))
                patient_intervals.sort()
    else:
        if verbose:
            print(f"  Bypassing Roles/RTTM filtering as requested (--skip_filtering)")

    # Load Transcript (Words)
    transcript_file = session_dir / "transcript.json"
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

    with open(transcript_file) as f:
        data = json.load(f)
    
    all_words = []
    if isinstance(data, dict):
        if "segments" in data:
            for seg in data["segments"]:
                if "words" in seg:
                    all_words.extend(seg["words"])
                else:
                    all_words.append({
                        "word": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"]
                    })
    
    # Filter words: Must overlap with a patient interval (if intervals exist and filtering enabled)
    patient_words = []
    
    if not skip_filtering and patient_intervals:
        for w in all_words:
            w_start = w["start"]
            w_end = w["end"]
            
            is_patient = False
            for (p_start, p_end) in patient_intervals:
                if max(w_start, p_start) < min(w_end, p_end):
                    is_patient = True
                    break
                if p_start > w_end:
                     break
            if is_patient:
                patient_words.append(w)
    else:
        # Fallback or Skip: keep all words
        patient_words = all_words

    if verbose:
        print(f"  Result: Keeping {len(patient_words)}/{len(all_words)} words")

    # 3. Segment text into 8-second windows
    embeddings = []
    for i in range(num_windows):
        start_time = i * 8.0
        end_time = (i + 1) * 8.0
        
        # Get patient words in this window
        window_words = [w for w in patient_words if start_time <= w['start'] < end_time]
        text = " ".join([w['word'].strip() for w in window_words])
        
        # Handle silent windows
        if not text.strip():
            text = "[SILENCE]"
        
        if verbose and i < 3:
            print(f"  Window {i}: '{text[:60]}...' ({len(window_words)} words)")
        
        # Extract embedding
        embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(embedding)
    
    # 4. Stack into tensor: [num_windows, 768]
    if embeddings:
        embeddings_tensor = torch.stack(embeddings)
    else:
        # Handle edge case: empty session
        embeddings_tensor = torch.zeros((num_windows, 768))

    # 5. Z-Score Normalization (Per-Session) to match Audio statistics
    # Standardize to Mean=0, Std=1
    mean = embeddings_tensor.mean(dim=0, keepdim=True)
    std = embeddings_tensor.std(dim=0, keepdim=True)
    embeddings_tensor = (embeddings_tensor - mean) / (std + 1e-8)
    
    if verbose:
        print(f"  Normalized: Mean={embeddings_tensor.mean():.4f}, Std={embeddings_tensor.std():.4f}")

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{session_id}_features.pt"  # Match audio naming convention
    torch.save(embeddings_tensor, output_file)
    
    return embeddings_tensor, num_windows, len(patient_words)

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B: Extract MentalRoBERTa embeddings with 8-second windowing"
    )
    parser.add_argument("--session_id", type=str, required=True, help="Session ID to process")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs",
        help="Path to Phase 1 rerun outputs"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_raw",
        help="Path to save text embeddings"
    )
    parser.add_argument("--model_name", type=str, default="mental/mental-roberta-base")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    
    args = parser.parse_args()
    
    # Add flag to skip filtering if source is already patient-only
    parser.add_argument("--skip_filtering", action="store_true", help="Skip RTTM/Role filtering (use all words)")
    # Re-parse to handle new argument
    args = parser.parse_args()

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("Using provided HF token for authentication")
    elif "HF_TOKEN" in os.environ:
        print("Using HF_TOKEN from environment")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, token=args.hf_token or os.environ.get("HF_TOKEN"))
    
    # Extract embeddings
    try:
        # Pass skip_filtering to the extraction function
        # (Need to update the function signature too, or just check args here)
        embeddings, num_windows, num_words = extract_text_embeddings(
            args.session_id, 
            args.input_dir, 
            args.output_dir, 
            model,
            verbose=args.verbose,
            skip_filtering=args.skip_filtering
        )
        
        print(f"✓ Session {args.session_id}: Shape={embeddings.shape}, Words={num_words}")
        
    except Exception as e:
        print(f"✗ Session {args.session_id}: {e}")
        raise

if __name__ == "__main__":
    main()
