import os
import json
import torch
import glob
import pandas as pd
import argparse

def verify_alignment(audio_features_dir, phase1_outputs_dir, output_csv="alignment_report.csv"):
    """
    Verifies temporal alignment by comparing:
    1. Duration from transcript.json (ground truth from Phase 1)
    2. Duration from audio features (frames * 0.02s)
    """
    print(f"🔍 Verifying Alignment...")
    print(f"   Audio Dir: {audio_features_dir}")
    print(f"   Phase 1 Output Dir: {phase1_outputs_dir}")
    
    results = []
    
    # Find all transcript.json files
    # Pattern: data/processed/phase1/session_outputs/{ID}/transcript.json
    transcript_files = glob.glob(os.path.join(phase1_outputs_dir, "*", "transcript.json"))
    
    print(f"   Found {len(transcript_files)} transcript files.")
    
    for tf in transcript_files:
        try:
            # Extract ID from path
            session_id = os.path.basename(os.path.dirname(tf))
            
            # Load Transcript Metadata
            with open(tf, 'r') as f:
                data = json.load(f)
                original_duration = data.get('duration', 0.0)
            
            # Load Audio Feature
            audio_path = os.path.join(audio_features_dir, f"{session_id}_features_norm.pt")
            
            if not os.path.exists(audio_path):
                results.append({
                    'id': session_id,
                    'original_duration': original_duration,
                    'feature_duration': 0,
                    'ratio': 0,
                    'diff_sec': -original_duration,
                    'status': 'MISSING_FEATURE'
                })
                continue
                
            # Load Tensor (CPU)
            audio_tensor = torch.load(audio_path, map_location='cpu')
            
            # Calculate Feature Duration
            # Assuming WavLM extracted at 20ms (0.02s) hop or frame length
            # Check dimensions: [T, 1024] or [Windows, T, 1024]?
            # Based on previous debugging, it was flattened to [T, 1024].
            # If it's strictly 20ms frame shift:
            
            # Count total frames
            if audio_tensor.ndim == 2:
                total_frames = audio_tensor.shape[0]
            elif audio_tensor.ndim == 3: # [Windows, Seq, Dim]
                total_frames = audio_tensor.shape[0] * audio_tensor.shape[1]
            else:
                total_frames = 0
                
            feature_duration = total_frames * 0.02
            
            # Compare
            ratio = feature_duration / original_duration if original_duration > 0 else 0
            diff = feature_duration - original_duration
            
            status = "OK"
            if abs(diff) > 5.0: # Allow 5 seconds variance (padding/trimming/header)
                status = "MISALIGNMENT_WARNING"
                
            results.append({
                'id': session_id,
                'original_duration': round(original_duration, 2),
                'feature_duration': round(feature_duration, 2),
                'frames': total_frames,
                'ratio': round(ratio, 4),
                'diff_sec': round(diff, 2),
                'status': status
            })
            
        except Exception as e:
            print(f"Error processing {tf}: {e}")
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save Report
    df.to_csv(output_csv, index=False)
    
    # Print Summary
    print("\n📊 Alignment Summary:")
    print(df['status'].value_counts())
    
    print("\n🔍 Top 5 Misalignments (Largest Diff):")
    print(df.sort_values(by='diff_sec', key=abs, ascending=False).head(5)[['id', 'original_duration', 'feature_duration', 'diff_sec', 'status']])
    
    print(f"\n✅ Report saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="/Users/umairtalib/DepressiVoice-LLM/data/03_features/audio_embeddings_clean/")
    parser.add_argument("--phase1_dir", default="/Users/umairtalib/DepressiVoice-LLM/data/processed/phase1/session_outputs/")
    parser.add_argument("--output", default="alignment_verification_report.csv")
    args = parser.parse_args()
    
    verify_alignment(args.audio_dir, args.phase1_dir, args.output)
