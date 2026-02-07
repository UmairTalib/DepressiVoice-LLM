import sys
import os

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

import argparse
import pandas as pd
import json
import time
from tqdm import tqdm
try:
    from src.phase5_llm.extract_v7_explanations import TransformerExplainer
except ImportError:
    from extract_v7_explanations import TransformerExplainer

OUTPUT_DIR = os.path.join(ROOT_DIR, "data/output/explanations/")
LOG_FILE = os.path.join(OUTPUT_DIR, "summary_log.csv")

def batch_generate(splits=['dev', 'test'], limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Explainer
    print("📋 Initializing TransformerExplainer (Loading Model & Labels)...")
    start_time = time.time()
    explainer = TransformerExplainer()
    print(f"✅ Explainer ready in {time.time() - start_time:.2f}s")
    
    # Load Labels
    labels_df = explainer.labels_df
    
    # Filter by Split
    target_df = labels_df[labels_df['split'].isin(splits)]
    
    if limit:
        target_df = target_df.head(limit)
        
    print(f"🚀 Starting Batch Generation for splits {splits}")
    print(f"📊 Total Participants: {len(target_df)}")
    
    print(f"🚀 Starting Batch Generation (Tier 1 Parallel Mode) for splits {splits}")
    print(f"📊 Total Participants: {len(target_df)}")
    
    results = []
    
    def process_patient(row_data):
        i, row = row_data
        pid = str(row['Participant_ID'])
        split = row['split']
        save_path = os.path.join(OUTPUT_DIR, f"v7_clinical_summary_{pid}.json")
        
        # Checkpoint: Skip if exists
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    existing_data = json.load(f)
                return {
                    "Participant_ID": pid,
                    "Split": split,
                    "Prob": existing_data['diagnostics']['depression_probability'],
                    "Severity": existing_data['diagnostics']['predicted_severity'],
                    "Risk_Level": existing_data['diagnostics']['risk_level'],
                    "Status": "Success (Cached)"
                }
            except Exception as e:
                print(f"⚠️ cached file corrupt for {pid}, re-processing: {e}")

        try:
            # Generate Report (Gemini Thinking happens here)
            report = explainer.run_explanation(pid)
            
            # Save JSON
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            print(f"✅ {pid} Done.")
            
            return {
                "Participant_ID": pid,
                "Split": split,
                "Prob": report['diagnostics']['depression_probability'],
                "Severity": report['diagnostics']['predicted_severity'],
                "Risk_Level": report['diagnostics']['risk_level'],
                "Status": "Success"
            }
            
        except FileNotFoundError:
            print(f"⚠️ Skipped {pid}: Feature/Transcript not found.")
            return {
                "Participant_ID": pid,
                "Split": split,
                "Status": "Missing Data"
            }
        except Exception as e:
            print(f"❌ Error {pid}: {e}")
            return {
                "Participant_ID": pid,
                "Status": f"Error: {str(e)[:50]}"
            }

    # Sequential Execution
    print(f"🚀 Processing {len(target_df)} participants sequentially...")
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        res = process_patient((idx, row))
        results.append(res)
        
        # Periodic save of log (every 5 patients)
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(LOG_FILE, index=False)

    # Save Log
    log_df = pd.DataFrame(results)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"\n✅ Batch Generation Complete. Log saved to {LOG_FILE}")
    print(log_df['Status'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients for testing")
    parser.add_argument("--splits", nargs="+", default=['dev', 'test'], help="Splits to process (train, dev, test)")
    args = parser.parse_args()
    
    batch_generate(splits=args.splits, limit=args.limit)
