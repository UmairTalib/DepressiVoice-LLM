#!/usr/bin/env python3
"""
Phase 1 Validation Suite - Comprehensive Quality Checks
Runs acoustic, linguistic, conversational, and alignment validations
"""

import argparse
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

def run_acoustic_validation(output_dir, sessions):
    """Run acoustic biometric validation"""
    print("\n=== ACOUSTIC VALIDATION ===")
    print("Validating speaker separation quality...")
    
    results = []
    margin_scores = []
    
    for session_dir in tqdm(sessions, desc="Acoustic validation"):
        roles_file = session_dir / "roles.json"
        
        if not roles_file.exists():
            continue
        
        with open(roles_file, 'r') as f:
            roles_data = json.load(f)
        
        margin = roles_data.get("separation_margin", 0)
        margin_scores.append(margin)
        
        results.append({
            "session_id": session_dir.name.replace("session_", ""),
            "separation_margin": margin,
            "ellie_similarity": max(roles_data.get("similarities", {}).values()),
            "patient_similarity": min(roles_data.get("similarities", {}).values())
        })
    
    # Calculate aggregate metrics
    margins = np.array(margin_scores)
    
    summary = {
        "total_sessions": len(results),
        "mean_separation_margin": float(np.mean(margins)),
        "median_separation_margin": float(np.median(margins)),
        "std_separation_margin": float(np.std(margins)),
        "min_separation_margin": float(np.min(margins)),
        "max_separation_margin": float(np.max(margins)),
        "sessions_below_threshold_01": int(np.sum(margins < 0.1)),
        "sessions_below_threshold_02": int(np.sum(margins < 0.2))
    }
    
    print(f"Mean Separation Margin: {summary['mean_separation_margin']:.4f}")
    print(f"Median: {summary['median_separation_margin']:.4f}")
    print(f"Sessions < 0.1 margin: {summary['sessions_below_threshold_01']}")
    
    return {"summary": summary, "per_session": results}

def run_linguistic_validation(output_dir, sessions):
    """Run linguistic contamination check"""
    print("\n=== LINGUISTIC VALIDATION ===")
    print("Checking for interviewer prompt leakage...")
    
    # High-risk prompts from original methodology
    ellie_lexicon = [
        "tell me about", "how are you", "what do you do", "how do you feel",
        "can you describe", "how would you", "what brings you", "how has that been"
    ]
    
    results = []
    total_segments = 0
    contaminated_segments = 0
    
    for session_dir in tqdm(sessions, desc="Linguistic validation"):
        transcript_file = session_dir / "transcript.json"
        
        if not transcript_file.exists():
            continue
        
        with open(transcript_file, 'r') as f:
            data = json.load(f)
        
        segments = data.get("segments", [])
        session_contaminated = 0
        
        for seg in segments:
            text = seg.get("text", "").lower()
            total_segments += 1
            
            for phrase in ellie_lexicon:
                if phrase in text:
                    session_contaminated += 1
                    contaminated_segments += 1
                    break
        
        results.append({
            "session_id": session_dir.name.replace("session_", ""),
            "total_segments": len(segments),
            "contaminated_segments": session_contaminated,
            "contamination_rate": session_contaminated / len(segments) if segments else 0
        })
    
    contamination_rate = contaminated_segments / total_segments if total_segments > 0 else 0
    
    summary = {
        "total_sessions": len(results),
        "total_segments": total_segments,
        "contaminated_segments": contaminated_segments,
        "contamination_rate": float(contamination_rate),
        "contamination_percentage": float(contamination_rate * 100)
    }
    
    print(f"Contamination Rate: {summary['contamination_percentage']:.2f}%")
    print(f"Contaminated Segments: {contaminated_segments} / {total_segments}")
    
    return {"summary": summary, "per_session": results}

def run_conversational_validation(output_dir, sessions):
    """Run conversational dynamics analysis"""
    print("\n=== CONVERSATIONAL DYNAMICS ===")
    print("Analyzing turn-taking patterns...")
    
    results = []
    turn_densities = []
    
    for session_dir in tqdm(sessions, desc="Conversational validation"):
        rttm_file = list(session_dir.glob("*.rttm"))
        
        if not rttm_file:
            continue
        
        rttm_file = rttm_file[0]
        
        # Count speaker turns
        with open(rttm_file, 'r') as f:
            lines = f.readlines()
        
        total_turns = len(lines)
        total_duration = sum(float(line.split()[4]) for line in lines)
        
        turn_density = total_turns / total_duration if total_duration > 0 else 0
        turn_densities.append(turn_density)
        
        results.append({
            "session_id": session_dir.name.replace("session_", ""),
            "total_turns": total_turns,
            "total_duration": total_duration,
            "turn_density": turn_density
        })
    
    densities = np.array(turn_densities)
    
    summary = {
        "total_sessions": len(results),
        "mean_turn_density": float(np.mean(densities)),
        "median_turn_density": float(np.median(densities)),
        "std_turn_density": float(np.std(densities))
    }
    
    print(f"Mean Turn Density: {summary['mean_turn_density']:.4f} turns/second")
    
    return {"summary": summary, "per_session": results}

def run_alignment_validation(output_dir, sessions):
    """Validate transcript-audio timestamp alignment"""
    print("\n=== ALIGNMENT VALIDATION ===")
    print("Checking transcript-audio synchronization...")
    
    results = []
    alignment_errors = 0
    
    for session_dir in tqdm(sessions, desc="Alignment validation"):
        transcript_file = session_dir / "transcript.json"
        patient_audio = session_dir / "patient_audio.wav"
        
        if not (transcript_file.exists() and patient_audio.exists()):
            continue
        
        # Simple check: verify transcript timestamps don't exceed audio duration
        with open(transcript_file, 'r') as f:
            data = json.load(f)
        
        duration = data.get("duration", 0)
        segments = data.get("segments", [])
        
        last_timestamp = max([seg.get("end", 0) for seg in segments]) if segments else 0
        
        is_aligned = last_timestamp <= duration + 1.0  # 1 second tolerance
        
        if not is_aligned:
            alignment_errors += 1
        
        results.append({
            "session_id": session_dir.name.replace("session_", ""),
            "audio_duration": duration,
            "last_transcript_timestamp": last_timestamp,
            "is_aligned": is_aligned
        })
    
    summary = {
        "total_sessions": len(results),
        "aligned_sessions": len(results) - alignment_errors,
        "misaligned_sessions": alignment_errors,
        "alignment_rate": float((len(results) - alignment_errors) / len(results)) if results else 0
    }
    
    print(f"Alignment Rate: {summary['alignment_rate']*100:.2f}%")
    print(f"Misaligned: {alignment_errors} / {len(results)}")
    
    return{"summary": summary, "per_session": results}

def save_results(all_results, output_dir):
    """Save validation results to multiple formats"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = output_dir / f"validation_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {json_file}")
    
    # Save summary CSV
    summary_file = output_dir / f"validation_summary_{timestamp}.csv"
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Validation Type", "Metric", "Value"])
        
        for val_type, data in all_results.items():
            if val_type == "metadata":
                continue
            for metric, value in data["summary"].items():
                writer.writerow([val_type, metric, value])
    
    print(f"Summary saved to: {summary_file}")
    
    return str(json_file), str(summary_file)

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Validation Suite")
    parser.add_argument("--output_dir", required=True, help="Phase 1 rerun outputs directory")
    parser.add_argument("--results_dir", default="validation_results", help="Validation results directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all session directories
    sessions = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])
    
    print(f"\n=== PHASE 1 VALIDATION SUITE ===")
    print(f"Output directory: {output_dir}")
    print(f"Total sessions found: {len(sessions)}")
    
    # Run all validations
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "total_sessions": len(sessions)
        },
        "acoustic": run_acoustic_validation(output_dir, sessions),
        "linguistic": run_linguistic_validation(output_dir, sessions),
        "conversational": run_conversational_validation(output_dir, sessions),
        "alignment": run_alignment_validation(output_dir, sessions)
    }
    
    # Save results
    save_results(all_results, results_dir)
    
    print("\n=== VALIDATION COMPLETE ===")
    
    return 0

if __name__ == "__main__":
    exit(main())
