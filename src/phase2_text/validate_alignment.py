#!/usr/bin/env python3
"""
Phase 2B Validation: Audio-Text Alignment Verification
Ensures text embeddings match audio features window-by-window.
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

def validate_alignment(audio_dir, text_dir, session_ids=None, verbose=False):
    """
    Validate that audio and text embeddings have matching window counts.
    
    Args:
        audio_dir: Path to audio_embeddings_normalized
        text_dir: Path to text_embeddings_raw
        session_ids: List of session IDs to check (None = all)
        verbose: Print per-session details
    
    Returns:
        dict: Validation results
    """
    audio_path = Path(audio_dir)
    text_path = Path(text_dir)
    
    # Get session IDs
    if session_ids is None:
        text_files = list(text_path.glob("*_embeddings.pt"))
        session_ids = [f.stem.replace("_embeddings", "") for f in text_files]
    
    results = {
        "total": len(session_ids),
        "passed": 0,
        "failed": 0,
        "mismatches": [],
        "missing_audio": [],
        "missing_text": [],
        "dimension_errors": []
    }
    
    print(f"Validating {len(session_ids)} sessions...")
    
    for sid in tqdm(session_ids):
        audio_file = audio_path / f"{sid}_features.pt"
        text_file = text_path / f"{sid}_embeddings.pt"
        
        # Check file existence
        if not audio_file.exists():
            results["missing_audio"].append(sid)
            results["failed"] += 1
            if verbose:
                print(f"✗ Session {sid}: Missing audio file")
            continue
        
        if not text_file.exists():
            results["missing_text"].append(sid)
            results["failed"] += 1
            if verbose:
                print(f"✗ Session {sid}: Missing text file")
            continue
        
        # Load and check shapes
        try:
            audio_feat = torch.load(audio_file)
            text_emb = torch.load(text_file)
            
            audio_windows = audio_feat.shape[0]
            text_windows = text_emb.shape[0]
            
            # Check window count match
            if audio_windows != text_windows:
                results["mismatches"].append({
                    "session_id": sid,
                    "audio_windows": audio_windows,
                    "text_windows": text_windows
                })
                results["failed"] += 1
                if verbose:
                    print(f"✗ Session {sid}: Window mismatch (audio={audio_windows}, text={text_windows})")
                continue
            
            # Check text embedding dimension
            if text_emb.shape[1] != 768:
                results["dimension_errors"].append({
                    "session_id": sid,
                    "expected": 768,
                    "actual": text_emb.shape[1]
                })
                results["failed"] += 1
                if verbose:
                    print(f"✗ Session {sid}: Wrong embedding dimension ({text_emb.shape[1]})")
                continue
            
            # Passed all checks
            results["passed"] += 1
            if verbose:
                print(f"✓ Session {sid}: {audio_windows} windows, aligned")
        
        except Exception as e:
            results["failed"] += 1
            if verbose:
                print(f"✗ Session {sid}: Error loading files - {e}")
    
    return results

def print_summary(results):
    """Print validation summary."""
    print("\n" + "="*50)
    print("ALIGNMENT VALIDATION SUMMARY")
    print("="*50)
    print(f"Total Sessions:    {results['total']}")
    print(f"Passed:            {results['passed']} ({results['passed']/results['total']*100:.1f}%)")
    print(f"Failed:            {results['failed']}")
    print()
    
    if results["mismatches"]:
        print(f"Window Mismatches: {len(results['mismatches'])}")
        for item in results["mismatches"][:5]:
            print(f"  - Session {item['session_id']}: audio={item['audio_windows']}, text={item['text_windows']}")
        if len(results["mismatches"]) > 5:
            print(f"  ... and {len(results['mismatches']) - 5} more")
    
    if results["missing_audio"]:
        print(f"Missing Audio:     {len(results['missing_audio'])}")
    
    if results["missing_text"]:
        print(f"Missing Text:      {len(results['missing_text'])}")
    
    if results["dimension_errors"]:
        print(f"Dimension Errors:  {len(results['dimension_errors'])}")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Validate audio-text alignment")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized",
        help="Path to audio embeddings"
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_raw",
        help="Path to text embeddings"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-session details")
    
    args = parser.parse_args()
    
    # Validate
    results = validate_alignment(args.audio_dir, args.text_dir, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Exit with error if any failures
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()
