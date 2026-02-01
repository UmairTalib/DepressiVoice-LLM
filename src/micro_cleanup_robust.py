
import os
import json
import csv
from pathlib import Path
import re

def micro_cleanup_robust():
    print("\n=== STARTING ROBUST MICRO-CLEANUP (V4) ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/cleanup_reports")
    
    targets = {
        "411": ["what do you mean"],
        "609": ["tell me about"],
        "684": ["when was the last time"]
    }
    
    manifest_rows = []
    
    for sid, phrases in targets.items():
        sdir = base_dir / sid
        txt_path = sdir / "cleaned_patient_text_v2.txt"
        
        if not txt_path.exists(): continue
            
        print(f"Processing {sid}...")
        
        # Load Full Text
        with open(txt_path, 'r') as f:
            content = f.read() # Keep case for restoration, but search lower
            
        content_lower = content.lower()
        original_words = len(content.split())
        
        # Robust Removal (Regex Sub)
        new_content = content
        hits_found = 0
        
        for phrase in phrases:
            # Case insensitive replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            # Check if exists
            if pattern.search(new_content):
                hits_found += 1
                # Replace with space to avoid merging adjacent words
                new_content = pattern.sub(" ", new_content)
        
        # Clean up double spaces
        new_content = re.sub(r'\s+', ' ', new_content).strip()
        final_words = len(new_content.split())
        
        # Save V4 TXT
        with open(sdir / "cleaned_patient_text_v4.txt", "w") as f:
            f.write(new_content)
            
        # Save V4 JSON (Wrapped as single segment since we broke alignment)
        v4_json = {
            "session_id": sid,
            "text": new_content,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 0.0, # Lost timing info in text-based cleaning
                    "text": new_content,
                    "avg_logprob": 0.0
                }
            ]
        }
        with open(sdir / "cleaned_patient_text_v4.json", "w") as f:
            json.dump(v4_json, f, indent=2)
            
        manifest_rows.append({
            "session_id": sid,
            "removed_segments": hits_found, # Count of phrases replaced
            "removed_text_sample": str(phrases),
            "words_before": original_words,
            "words_after": final_words,
            "percent_loss": (1 - final_words/original_words)*100 if original_words > 0 else 0
        })

    # Update Report
    with open(report_dir / "micro_cleanup_report.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
        
    print(json.dumps(manifest_rows, indent=2))

if __name__ == "__main__":
    micro_cleanup_robust()
