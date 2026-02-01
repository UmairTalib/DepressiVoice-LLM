
import json
import pandas as pd
from pathlib import Path

def consolidate_results():
    root = Path("/Users/umairtalib/DepressiVoice-LLM/validation_final_bundle")
    
    # Load Acoustic
    with open(root / "eer_report.json", 'r') as f:
        acoustic = json.load(f)
        
    # Load Linguistic
    with open(root / "linguistic_purity_report.json", 'r') as f:
        linguistic = json.load(f)
        
    # Load Conversational
    with open(root / "conversational_metrics.json", 'r') as f:
        conversational = json.load(f)
        
    master_summary = {
        "acoustic": {
            "roc_auc": acoustic["roc_auc"],
            "eer": acoustic["eer"],
            "margin_mean": acoustic["margin_bootstrap"]["mean"],
            "margin_ci": [acoustic["margin_bootstrap"]["lower_95_ci"], acoustic["margin_bootstrap"]["upper_95_ci"]]
        },
        "linguistic": {
            "contamination_rate": linguistic["contamination_rate_percent"],
            "top_phrases": linguistic["top_matched_phrases"]
        },
        "conversational": {
            "avg_turn_density_tpm": conversational["avg_turn_density_tpm"],
            "outlier_count": conversational["outlier_count"],
            "avg_duration": conversational["avg_turn_duration"]
        }
    }
    
    with open(root / "master_validation_summary.json", "w") as f:
        json.dump(master_summary, f, indent=2)
    
    print("Master summary consolidated.")

if __name__ == "__main__":
    consolidate_results()
