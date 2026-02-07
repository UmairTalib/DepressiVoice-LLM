import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.phase4_temporal.models import ImprovedLSTMAggregator

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================

# Use dynamic root detection for path flexibility
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths (Updated to be relative to project root)
MODEL_PATH = os.path.join(ROOT_DIR, "experiments/phase4_temporal/best_hybrid_lstm.pth")
LABELS_PATH = os.path.join(ROOT_DIR, "data/01_raw/detailed_labels.csv")
FEATURE_DIR = os.path.join(ROOT_DIR, "data/phase4_features/")
OUTPUT_CSV = os.path.join(ROOT_DIR, "experiments/phase5_validation_results.csv")

THRESHOLD = 0.05  # The Phase 5 Calibration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_robust_model(model_path):
    """Loads the model with key-mapping to handle attention vs attention_net mismatch."""
    model = ImprovedLSTMAggregator(input_dim=256, hidden_dim=64, num_layers=1, dropout=0.5).to(DEVICE)
    
    if not os.path.exists(model_path):
        # Fallback to alternative name if exists
        alt_path = model_path.replace("best_hybrid_lstm.pth", "best_augmented_focal_lstm.pth")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"❌ Model weights not found at {model_path}")
            
    print(f"🔄 Loading Champion Model: {model_path}")
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Robust key mapping (Handles attention vs attention_net mismatch)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("attention_net.") and not hasattr(model, 'attention_net'):
            new_state_dict[k.replace("attention_net.", "attention.")] = v
        elif k.startswith("attention.") and not hasattr(model, 'attention'):
            new_state_dict[k.replace("attention.", "attention_net.")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def load_robust_features(feat_path):
    """Loads features and extracts 'fused_embedding' if saved as a dict."""
    features = torch.load(feat_path, map_location='cpu')
    
    if isinstance(features, dict):
        if 'fused_embedding' in features:
            features = features['fused_embedding']
        elif 'features' in features:
            features = features['features']
            
    if hasattr(features, 'shape') and len(features.shape) == 2:
        features = features.unsqueeze(0)
    return features.to(DEVICE)

def run_validation():
    print(f"🚀 Starting Phase 5 Diagnostic Validation (Threshold: {THRESHOLD})")
    
    # 1. Load Ground Truth
    actual_labels_path = LABELS_PATH
    if not os.path.exists(actual_labels_path):
        # Fallback for typo in filename
        actual_labels_path = LABELS_PATH.replace("detailed_labels.csv", "detailed_lables.csv")
    
    if not os.path.exists(actual_labels_path):
        raise FileNotFoundError(f"❌ Labels file not found at {LABELS_PATH}")
        
    df = pd.read_csv(actual_labels_path)
    print(f"📂 Loaded {len(df)} patients from metadata.")
        
    # Filter for Dev set only (if split column exists)
    if 'split' in df.columns:
        df = df[df['split'] == 'dev']
        print(f"🔍 Filtering to 'dev' set: {len(df)} patients.")

    # 2. Load Model
    model = load_robust_model(MODEL_PATH)

    results = []
    y_true = []
    y_pred = []

    # 3. Batch Inference
    print("⚡ Running Inference...")
    for _, row in df.iterrows():
        pid = str(row['Participant'])
        label = int(row['Depression_label']) # 0 or 1
        
        # Construct path (Handle both with and without _P suffix)
        feat_path = os.path.join(FEATURE_DIR, f"{pid}_fused.pt")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(FEATURE_DIR, f"{pid}_P_fused.pt")
            
        if os.path.exists(feat_path):
            # Load & Predict
            feats = load_robust_features(feat_path)
            with torch.no_grad():
                logits, _ = model(feats)
                prob = torch.sigmoid(logits).item()
            
            # Phase 5 Decision Logic
            pred_label = 1 if prob >= THRESHOLD else 0
            
            y_true.append(label)
            y_pred.append(pred_label)
            
            results.append({
                "Patient": pid,
                "Ground_Truth": label,
                "AI_Score": prob,
                "AI_Risk_Label": "HIGH" if pred_label == 1 else "LOW",
                "Correct": label == pred_label
            })
        else:
            # Skip if feature file is missing for this patient (common in E-DAIC)
            continue

    if not y_true:
        print("❌ No valid feature files found for the given patients. Check FEATURE_DIR.")
        return

    # 4. Generate Report
    print("\n" + "="*40)
    print("📊 PHASE 5 DIAGNOSTIC ACCURACY REPORT")
    print("="*40)
    
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Handle division by zero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Total Patients Analyzed: {len(y_true)}")
    print(f"ACCURACY: {acc:.4f}")
    print(f"SENSITIVITY (Recall): {sensitivity:.4f} (Target: High)")
    print(f"SPECIFICITY: {specificity:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"TP (Depressed detected): {tp}")
    print(f"FN (Depressed missed):   {fn}  <-- CRITICAL STAT")
    print(f"FP (False Alarms):       {fp}")
    print(f"TN (Healthy Correct):    {tn}")
    print("-" * 40)
    
    # Save detailed CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"📝 Detailed predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_validation()
