#!/usr/bin/env python3
"""
Phase 1c: SOTA Transcription Pipeline (Parallel Sharding Enabled)
Uses Faster-Whisper (Large-v3) to transcribe extracted patient audio.
"""

import os
import json
import argparse
import time
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm

def setup_model(model_size="large-v3", device="cuda", compute_type="float16"):
    print(f"Loading Faster-Whisper model: {model_size} ({device}, {compute_type})...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to int8 if float16 fails (common on older GPUs)
        if compute_type == "float16":
            print("Retrying with int8_float16...")
            return WhisperModel(model_size, device=device, compute_type="int8_float16")
        raise e

def transcribe_file(model, audio_path, output_dir):
    start_time = time.time()
    
    # Transcribe (SOTA settings)
    segments, info = model.transcribe(str(audio_path), beam_size=5, language="en")
    
    transcript_text = ""
    transcript_segments = []
    
    for segment in segments:
        transcript_text += segment.text + " "
        transcript_segments.append({
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob
        })
        
    duration = time.time() - start_time
    
    # Save Results
    result = {
        "session_id": audio_path.stem.replace("patient_audio_", ""),
        "audio_file": audio_path.name,
        "duration": info.duration,
        "language": info.language,
        "transcription_time": duration,
        "text": transcript_text.strip(),
        "segments": transcript_segments
    }
    
    # Write JSON (Rich Metadata)
    json_path = output_dir / "transcript.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
        
    # Write TXT (Plain text for quick review)
    txt_path = output_dir / "transcript.txt"
    with open(txt_path, "w") as f:
        f.write(transcript_text.strip())
        
    return True

def process_sharded_sessions(base_dir, model, shard_id, num_shards):
    # Search for patient_audio_*.wav recursively
    print(f"Scanning for audio in: {base_dir}")
    all_files = sorted(list(base_dir.rglob("patient_audio_*.wav")))
    
    # Apply Sharding
    my_files = all_files[shard_id::num_shards]
    
    print(f"Total Files: {len(all_files)}")
    print(f"My Shard ({shard_id}/{num_shards}): {len(my_files)} files")
    
    count = 0
    skipped = 0
    errors = 0
    
    for audio_path in tqdm(my_files, desc=f"Shard {shard_id}"):
        session_dir = audio_path.parent
        
        # Check if done
        if (session_dir / "transcript.json").exists():
            skipped += 1
            print(f"Skipping {audio_path.name} (Already done)")
            continue
            
        try:
            print(f"Processing: {audio_path.name}")
            transcribe_file(model, audio_path, session_dir)
            count += 1
        except Exception as e:
            print(f"FAILED: {audio_path.name} - {e}")
            errors += 1
            
    print(f"\n=== SHARD {shard_id} COMPLETE ===")
    print(f"Processed: {count}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        exit(1)
        
    model = setup_model(device=args.device)
    process_sharded_sessions(Path(args.input_dir), model, args.shard_id, args.num_shards)
