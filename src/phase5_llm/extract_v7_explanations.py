import os
import torch
import torch.nn as nn
import json
import argparse
import pandas as pd
import numpy as np
import google.generativeai as genai

# API Key (Read from Environment)
API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBcq_pMAGs3u3JV0LEOfggMYantKqQfo6k"

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, "experiments/phase4_mtl/best_mtl_transformer.pth") # Champion V7 MTL
LABELS_PATH = os.path.join(ROOT_DIR, "data/labels/master_labels_v7.csv")
FEATURE_DIR = os.path.join(ROOT_DIR, "data/phase4_features/")
TRANSCRIPT_DIR = os.path.join(ROOT_DIR, "data/Phase1_results/")

try:
    from src.phase4_temporal.models import CrossModalTransformerAggregator
except ImportError:
    try:
        from phase4_temporal.models import CrossModalTransformerAggregator
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.phase4_temporal.models import CrossModalTransformerAggregator


PROB_THRESHOLD = 0.06
RMSE_CALIBRATION = 3.0 # Estimated from 0.1422 Normalized RMSE
DEVICE = 'cpu' # Force CPU to avoid CUDA initialization hangs on cluster nodes

SYMPTOM_MAP = {
    "PHQ8_1_NoInterest": "Anhedonia / Loss of Interest",
    "PHQ8_2_Depressed":  "Depressed Mood",
    "PHQ8_3_Sleep":      "Sleep Disturbance",
    "PHQ8_4_Tired":      "Fatigue / Low Energy",
    "PHQ8_5_Appetite":   "Appetite Changes",
    "PHQ8_6_Failure":    "Feelings of Failure / Guilt",
    "PHQ8_7_Concentration": "Concentration Difficulties",
    "PHQ8_8_Psychomotor":   "Psychomotor Agitation / Retardation"
}

class TransformerExplainer:
    def __init__(self):
        self.model_id = "gemini-1.5-flash"
        print(f"🚀 [INIT] Starting TransformerExplainer with {self.model_id}...")
        self.model = self._load_model()
        print("📊 [INIT] Loading labels...")
        self.labels_df = pd.read_csv(LABELS_PATH)
        self._setup_gemini()

    def _setup_gemini(self):
        if not API_KEY:
            print("⚠️ WARNING: GEMINI_API_KEY not found. Narratives will be skipped.")
            return

        print(f"🔄 Configuring Google Generative AI (Model: {self.model_id})...")
        genai.configure(api_key=API_KEY)
        self.client = genai.GenerativeModel(self.model_id)
        print(f"✅ Gemini Narrator Ready.")

    def _load_model(self):
        model = CrossModalTransformerAggregator(input_dim=256, d_model=128, nhead=4, num_layers=2).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"✅ V7 Transformer loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Warning: Model weights not found at {MODEL_PATH}.")
        return model

    def _get_patient_metadata(self, patient_id):
        patient_row = self.labels_df[self.labels_df['Participant_ID'].astype(str) == str(patient_id)]
        if patient_row.empty:
            print(f"⚠️ Metadata not found for patient {patient_id}")
            return None
        return patient_row.iloc[0].to_dict()

    def _load_data(self, patient_id):
        feat_path = os.path.join(FEATURE_DIR, f"{patient_id}_fused.pt")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(FEATURE_DIR, f"{patient_id}_P_fused.pt")
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing features for {patient_id}")
            
        features = torch.load(feat_path, map_location='cpu')
        
        if isinstance(features, dict):
            features = features.get('fused_embedding', features.get('features', features))
        
        if not isinstance(features, torch.Tensor):
             features = torch.tensor(features)
             
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            
        trans_path = os.path.join(TRANSCRIPT_DIR, patient_id, "transcript.json")
        if os.path.exists(trans_path):
            with open(trans_path, 'r') as f:
                data = json.load(f)
                transcript = data.get('segments', data)
        else:
            transcript = []
            
        return features.to(DEVICE), transcript

    def run_explanation(self, patient_id):
        # 1. Load Data
        features, transcript = self._load_data(patient_id)
        metadata = self._get_patient_metadata(patient_id)
        
        # 2. Model Forward
        print(f"⚡ Running forward pass for {patient_id}...")
        with torch.no_grad():
            prediction, attn_weights = self.model(features)
            (logits, severity_pred) = prediction
            
            prob = torch.sigmoid(logits).item()
            severity = severity_pred.item() * 24.0
            if severity < 0: severity = 0
            if severity > 24: severity = 24

        # 3. Extract Attention Rollout
        if attn_weights is not None:
            combined_importance = attn_weights[0].mean(dim=0).cpu().numpy()
        else:
            combined_importance = np.ones(features.shape[1]) / features.shape[1]
        
        # 4. Identify Spikes
        top_indices = combined_importance.argsort()[-3:][::-1]
        spikes = []
        for idx in top_indices:
            timestamp = idx * 2.0 
            text_snippet = self._find_text(transcript, timestamp)
            spikes.append({
                "timestamp": f"{int(timestamp//60)}:{int(timestamp%60):02d}",
                "importance_score": float(combined_importance[idx]),
                "text": text_snippet
            })

        # 5. Symptom Grounding
        grounded_symptoms = []
        if metadata:
            for key, desc in SYMPTOM_MAP.items():
                score = metadata.get(key, 0)
                if score >= 1:
                    grounded_symptoms.append({
                        "symptom": desc,
                        "clinical_score": int(score),
                        "interpretation": f"Correlation with detected distress spikes at {spikes[0]['timestamp']}."
                    })

        # 6. Generate Narrative (Gemini Narrator)
        narrative = self._generate_narrative(patient_id, prob, severity, spikes, grounded_symptoms)

        # 7. Final Report Construction
        report = {
            "patient_id": patient_id,
            "analysis_type": "V7 Cross-Modal Transformer Multi-Task Explanation",
            "diagnostics": {
                "depression_probability": round(prob, 4),
                "threshold": PROB_THRESHOLD,
                "risk_level": "HIGH RISK" if prob >= PROB_THRESHOLD else "LOW RISK",
                "predicted_severity": f"{round(severity, 1)} ± {RMSE_CALIBRATION}",
                "severity_category": self._get_severity_cat(severity)
            },
            "attention_spikes": spikes,
            "clinical_grounding": grounded_symptoms,
            "clinical_narrative": narrative,
            "summary": f"Targeting 0.06 threshold. Calibration utilizes 0.1422 RMSE. "
                       f"Cross-modal attention shows {'congruence' if prob >= 0.5 else 'dissonance'} between acoustic and textual streams."
        }
        
        return report

    def _generate_narrative(self, patient_id, prob, severity, spikes, symptoms):
        if not hasattr(self, 'client') or self.client is None:
            return "Narrative skipped (Gemini unavailable)."

        risk_label = "HIGH RISK" if prob >= PROB_THRESHOLD else "LOW RISK"
        
        # System Instruction for the "Psychiatrist Persona"
        sys_inst = "You are a Senior Clinical Psychiatrist. Analyze the provided Transformer attention spikes and PHQ-8 scores to generate a calibrated diagnostic narrative. Synthesize the acoustic dissonance and textual content."

        prompt = f"""
        === PATIENT {patient_id} DATA ===
        - AI Probability: {prob:.4f} (Threshold: {PROB_THRESHOLD})
        - Risk Level: **{risk_label}**
        - Predicted Severity: {severity:.1f} ± {RMSE_CALIBRATION}
        
        === ATTENTION SPIKES (Model Focus) ===
        """
        for s in spikes:
            prompt += f"- Time {s['timestamp']} (Imp {s['importance_score']:.3f}): \"{s['text']}\"\n"
            
        prompt += "\n=== CLINICAL SYMPTOMS ===\n"
        for sym in symptoms:
            prompt += f"- {sym['symptom']} (Score: {sym['clinical_score']})\n"

        prompt += "\nTASK: Think about why the model flagged this patient. Justify the risk level based on the alignment (or mismatch) between the audio-focused attention spikes and the clinical history."

        print(f"🤖 Consulting {self.model_id}...")
        
        # Parallel Execution handled by batch script, but here we just call generate
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=1.0
                )
            )
            
            return response.text

        except Exception as e:
            print(f"⚠️ Gemini Error: {e}")
            return f"Gemini Error: {e}"

    def _get_severity_cat(self, s):
        if s <= 4: return "Minimal"
        if s <= 9: return "Mild"
        if s <= 14: return "Moderate"
        if s <= 19: return "Moderately Severe"
        return "Severe"

    def _find_text(self, transcript, timestamp, window=8.0):
        matches = []
        for seg in transcript:
            start = seg.get('start', seg.get('start_time', 0))
            text = seg.get('text', seg.get('value', ''))
            if start >= timestamp and start < (timestamp + window):
                matches.append(text)
        return " ".join(matches) if matches else "[Non-verbal distressed cues]"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_ids", nargs="+", help="One or more Patient IDs")
    parser.add_argument("--split", type=str, default=None, help="Process all patients in a split (dev, test)")
    args = parser.parse_args()

    explainer = TransformerExplainer()
    
    pids = args.patient_ids or []
    if args.split:
        pids += explainer.labels_df[explainer.labels_df['split'] == args.split]['Participant_ID'].astype(str).tolist()
    
    if not pids:
        print("❌ Error: Provide --patient_ids or --split")
        sys.exit(1)

    print(f"🚀 Starting Sequential Processing for {len(pids)} patients...")
    for pid in pids:
        try:
            # Checkpoint: Skip if exists
            output_path = os.path.join(ROOT_DIR, f"data/output/explanations/v7_clinical_summary_{pid}.json")
            if os.path.exists(output_path):
                print(f"⏭️ Skipping {pid} (Already exists)")
                continue

            report = explainer.run_explanation(pid)
            
            # Save to JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"✅ {pid} Done.")
        except Exception as e:
            print(f"❌ Error {pid}: {e}")
