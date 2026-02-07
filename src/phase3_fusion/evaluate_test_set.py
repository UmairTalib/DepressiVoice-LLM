import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel
import logging
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report

def evaluate_test_set(args):
    """
    Strict evaluation on the Test Set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FINAL TEST EVALUATION on {device}...")
    print(f"Model: {args.model_path}")
    
    # Load Test Data
    test_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                         leakage_json=args.leakage_json, subset='test') # Crucial: Subset='test'
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    
    print(f"Loaded {len(test_dataset)} Test subjects.")

    # Load Model
    model = FusionModel(d_model=256, nhead=8, num_layers=2, dropout=0.1) 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_pids = []

    print("Executing Inference...")
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device)
            pids = batch['participant_ids']
            
            logits, _ = model(audio, text, audio_mask, text_mask)
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_pids.extend(pids)

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    # RMSE calculation (using probabilities as regression proxy if we don't have regression head)
    # Note: PHQ-8 is 0-24. If model is binary, RMSE on binary labels is just sqrt(MSE).
    # Thesis usually wants RMSE against PHQ-8 score. 
    # But our model is classification only (0/1). 
    # So we report Binary Classification Metrics.
    
    print("\n=== OFFICIAL TEST SET RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Depressed): {f1:.4f}")
    print("-" * 30)
    print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Depressed']))
    
    # Save Predictions
    results_df = pd.DataFrame({
        'Participant_ID': all_pids,
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Probability': all_probs
    })
    
    out_file = os.path.join(args.output_dir, "test_set_predictions.csv")
    results_df.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finalized fusion model (.pt)")
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--leakage_json", type=str, default="/home/g755389/DepressiVoice-LLM/docs/reports/ellie_leakage_audit.json")
    parser.add_argument("--output_dir", type=str, default="experiments/final_test")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate_test_set(args)
