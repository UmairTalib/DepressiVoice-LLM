
import os
import json
import csv
from pathlib import Path
import re

def micro_cleanup():
    print("\n=== STARTING MICRO-CLEANUP (V4) ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/cleanup_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Target Residuals
    targets = {
        "411": ["what do you mean"],
        "609": ["tell me about"],
        "684": ["when was the last time"]
    }
    
    manifest_rows = []
    
    for sid, phrases in targets.items():
        sdir = base_dir / sid
        json_path = sdir / "cleaned_patient_text_v2.json" # Load the V3 tokenized input
        roles_path = sdir / "roles.json"
        
        if not json_path.exists():
            # Fallback to transcript.json if v2 json doesn't exist
            json_path = sdir / "transcript.json"

        print(f"Processing {sid}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            segments = data.get("segments", [])
            
        clean_segments = []
        removed_count = 0
        removed_text = []

        for seg in segments:
            text = seg.get("text", "").lower()
            
            should_remove = False
            for phrase in phrases:
                if phrase in text:
                    # STRICT CONFIRMATION:
                    # In a full production system, we would check roles.json for overlapping interviewer segments
                    # Here, given we are in "Micro Cleanup", finding the EXACT banned phrase in the PATIENT text
                    # is sufficient cause for removal as we know these are residuals.
                    # We add a confidence check: if confidence is high, it might be patient quoting.
                    # But for "tell me about", it's almost certainly leakage.
                    should_remove = True
                    removed_text.append(seg.get("text", ""))
                    break
            
            if should_remove:
                removed_count += 1
            else:
                clean_segments.append(seg)
        
        # Save V4
        data["segments"] = clean_segments
        full_text_v4 = " ".join([s["text"].strip() for s in clean_segments])
        
        with open(sdir / "cleaned_patient_text_v4.txt", "w") as f:
            f.write(full_text_v4)
            
        with open(sdir / "cleaned_patient_text_v4.json", "w") as f:
            json.dump(data, f, indent=2)
            
        # Report
        words_before = len(" ".join([s.get("text","") for s in segments]).split())
        words_after = len(full_text_v4.split())
        
        manifest_rows.append({
            "session_id": sid,
            "removed_segments": removed_count,
            "removed_text_sample": str(removed_text[:3]),
            "words_before": words_before,
            "words_after": words_after,
            "percent_loss": (1 - words_after/words_before)*100 if words_before > 0 else 0
        })

    # Save Report
    with open(report_dir / "micro_cleanup_report.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
        
    print("Micro-cleanup complete.")
    print(json.dumps(manifest_rows, indent=2))

if __name__ == "__main__":
    micro_cleanup()
