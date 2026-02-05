#!/usr/bin/env python3
"""
Phase 1 Rerun Validation Suite
Validates all 275 sessions for quality, completeness, and correctness
"""

import json
import argparse
from pathlib import Path
import soundfile as sf
from collections import defaultdict
import csv

def validate_all_sessions(output_dir):
    """Validate all sessions"""
    
    print(f"\n{'='*70}")
    print(f"PHASE 1 RERUN - VALIDATION SUITE")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    session_dirs = sorted(output_path.glob("session_*"))
    
    results = {
        "total_sessions": 0,
        "complete": 0,
        "missing_files": 0,
        "quality_flags": 0,
        "rescue_sessions": 0,
        "confidence_distribution": defaultdict(int),
        "contamination_stats": [],
        "duration_stats": [],
        "issues": []
    }
    
    session_details = []
    
    print(f"Found {len(session_dirs)} session directories\n")
    print("Validating...")
    
    for session_dir in session_dirs:
        session_id = session_dir.name.replace("session_", "")
        results["total_sessions"] += 1
        
        # Check required files
        required_files = {
            "patient_audio": session_dir / "patient_audio.wav",
            "transcript": session_dir / "transcript.json",
            "patient_text": session_dir / "patient_text.txt"
        }
        
        optional_files = {
            "roles": session_dir / "roles.json",
            "rttm": session_dir / f"{session_id}.rttm",
            "quality_flag": session_dir / "quality_flag.json",
            "rescue_metadata": session_dir / "rescue_metadata.json"
        }
        
        # Check completeness
        missing = []
        for name, path in required_files.items():
            if not path.exists():
                missing.append(name)
        
        if missing:
            results["missing_files"] += 1
            results["issues"].append(f"Session {session_id}: Missing {', '.join(missing)}")
            continue
        
        # Load data
        with open(required_files["transcript"], 'r') as f:
            transcript = json.load(f)
        
        with open(required_files["patient_text"], 'r') as f:
            text = f.read()
        
        audio_info = sf.info(required_files["patient_audio"])
        
        # Check for quality flag
        has_quality_flag = optional_files["quality_flag"].exists()
        has_rescue = optional_files["rescue_metadata"].exists()
        
        if has_quality_flag:
            results["quality_flags"] += 1
        
        if has_rescue:
            results["rescue_sessions"] += 1
        
        # Get confidence if available
        confidence = "UNKNOWN"
        margin = 0
        if optional_files["roles"].exists():
            with open(optional_files["roles"], 'r') as f:
                roles = json.load(f)
                confidence = roles.get("confidence", "UNKNOWN")
                margin = roles.get("separation_margin", 0)
        
        results["confidence_distribution"][confidence] += 1
        
        # Check contamination
        segments = transcript.get("segments", [])
        ellie_phrases = ["tell me", "how are you", "what do you", "can you describe"]
        
        contaminated = 0
        for seg in segments:
            text_lower = seg.get("text", "").lower()
            if any(phrase in text_lower for phrase in ellie_phrases):
                contaminated += 1
        
        contamination_rate = (contaminated / len(segments) * 100) if segments else 0
        
        # Store details
        session_detail = {
            "session_id": session_id,
            "complete": True,
            "audio_duration": audio_info.duration,
            "segments": len(segments),
            "contamination_rate": contamination_rate,
            "confidence": confidence,
            "margin": margin,
            "quality_flag": has_quality_flag,
            "rescue": has_rescue,
            "language": transcript.get("language", "unknown")
        }
        
        session_details.append(session_detail)
        results["complete"] += 1
        results["contamination_stats"].append(contamination_rate)
        results["duration_stats"].append(audio_info.duration)
        
        # Progress indicator
        if results["total_sessions"] % 25 == 0:
            print(f"  Processed {results['total_sessions']} sessions...")
    
    # Calculate statistics
    if results["contamination_stats"]:
        avg_contamination = sum(results["contamination_stats"]) / len(results["contamination_stats"])
        max_contamination = max(results["contamination_stats"])
        
        avg_duration = sum(results["duration_stats"]) / len(results["duration_stats"])
        total_duration_hours = sum(results["duration_stats"]) / 3600
    else:
        avg_contamination = 0
        max_contamination = 0
        avg_duration = 0
        total_duration_hours = 0
    
    # Print report
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"📊 COMPLETENESS:")
    print(f"  Total Sessions:          {results['total_sessions']}")
    print(f"  Complete:                {results['complete']} ({results['complete']/results['total_sessions']*100:.1f}%)")
    print(f"  Missing Files:           {results['missing_files']}")
    print(f"  Quality Flags:           {results['quality_flags']}")
    print(f"  Rescue Sessions:         {results['rescue_sessions']}\n")
    
    print(f"🎯 CONFIDENCE DISTRIBUTION:")
    for conf, count in sorted(results["confidence_distribution"].items()):
        print(f"  {conf:12s}: {count:3d} sessions ({count/results['complete']*100:5.1f}%)")
    print()
    
    print(f"📝 TRANSCRIPTION QUALITY:")
    print(f"  Avg Contamination:       {avg_contamination:.2f}%")
    print(f"  Max Contamination:       {max_contamination:.2f}%")
    print(f"  Avg Duration:            {avg_duration:.1f} seconds")
    print(f"  Total Audio:             {total_duration_hours:.1f} hours\n")
    
    if results["issues"]:
        print(f"⚠️  ISSUES FOUND:")
        for issue in results["issues"]:
            print(f"  • {issue}")
        print()
    
    # Save results
    results_file = output_path / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {k: v for k, v in results.items() if k not in ["contamination_stats", "duration_stats"]},
            "statistics": {
                "avg_contamination": avg_contamination,
                "max_contamination": max_contamination,
                "avg_duration": avg_duration,
                "total_duration_hours": total_duration_hours
            }
        }, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")
    
    # Save CSV
    csv_file = output_path / "validation_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=session_details[0].keys())
        writer.writeheader()
        writer.writerows(session_details)
    
    print(f"✅ CSV saved: {csv_file}")
    
    print(f"\n{'='*70}\n")
    
    # Overall status
    if results["complete"] == results["total_sessions"] and not results["issues"]:
        print("🎉 VALIDATION PASSED - All sessions complete and valid!")
        return 0
    else:
        print("⚠️  VALIDATION WARNINGS - Review issues above")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Rerun Validation")
    parser.add_argument("--output_dir", 
                       default="/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    exit_code = validate_all_sessions(args.output_dir)
    return exit_code

if __name__ == "__main__":
    import sys
    sys.exit(main())
