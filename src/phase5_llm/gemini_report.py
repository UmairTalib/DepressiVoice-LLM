import os
import torch
import json
import argparse
import numpy as np
import google.generativeai as genai
from src.phase4_temporal.models import ImprovedLSTMAggregator

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================

# 1. API Key (Read from Environment)
# Run in terminal: export GEMINI_API_KEY="AIzaSy..."
API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Paths (Dynamic Root Detection)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_WEIGHTS = os.path.join(ROOT_DIR, "experiments/phase4_temporal/best_hybrid_lstm.pth")
FEATURE_DIR   = os.path.join(ROOT_DIR, "data/phase4_features/")
TRANSCRIPT_DIR = os.path.join(ROOT_DIR, "data/Phase1_results/")

# 3. Model Parameters
INPUT_DIM = 256
HIDDEN_DIM = 64
THRESHOLD = 0.05  # Calibrated for High Recall
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 🧠 CLINICAL AI AGENT
# ==========================================

class ClinicalAgent:
    def __init__(self):
        self._setup_gemini()
        self._load_model()

    def _setup_gemini(self):
        if not API_KEY:
            print("⚠️  WARNING: GEMINI_API_KEY not found in environment.")
            print("   Please run: export GEMINI_API_KEY='your_key_here'")
            return
        genai.configure(api_key=API_KEY)
        
        # Robust model selection
        model_options = ['gemini-2.0-flash-lite', 'gemini-flash-latest', 'gemini-2.0-flash', 'gemini-pro-latest']
        self.llm = None
        for model_id in model_options:
            try:
                self.llm = genai.GenerativeModel(model_id)
                # Test the model with a tiny request
                print(f"🔄 Testing Gemini model: {model_id}")
                self.llm.generate_content("Ping", generation_config={"max_output_tokens": 1})
                print(f"✅ Using Gemini model: {model_id}")
                break
            except Exception as e:
                print(f"⚠️ Model {model_id} failed: {e}")
                self.llm = None
                
        if not self.llm:
             print("❌ Failed to initialize any Gemini model.")

    def _load_model(self):
        print(f"🔄 Loading Champion Model: {MODEL_WEIGHTS}")
        self.model = ImprovedLSTMAggregator(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=1,
            dropout=0.5
        ).to(DEVICE)

        if os.path.exists(MODEL_WEIGHTS):
            state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
            
            # Robust key mapping (Handles attention vs attention_net mismatch)
            new_state_dict = {}
            for k, v in state_dict.items():
                # If model expects attention but weights have attention_net
                if k.startswith("attention_net.") and not hasattr(self.model, 'attention_net'):
                    new_state_dict[k.replace("attention_net.", "attention.")] = v
                # If model expects attention_net but weights have attention
                elif k.startswith("attention.") and not hasattr(self.model, 'attention'):
                    new_state_dict[k.replace("attention.", "attention_net.")] = v
                else:
                    new_state_dict[k] = v
                    
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            print("✅ Model loaded successfully.")
        else:
            raise FileNotFoundError(f"❌ Model weights not found at {MODEL_WEIGHTS}")

    def analyze_patient(self, patient_id):
        """Full pipeline: Load Data -> Inference -> LLM Report"""
        
        # 1. Load Data
        features, transcript = self._load_data(patient_id)
        
        # 2. Run Deep Learning Inference
        features = features.to(DEVICE)
        with torch.no_grad():
            logits, attn_weights = self.model(features)
            risk_score = torch.sigmoid(logits).item()
            
        # 3. Extract Evidence (Attention Spikes)
        evidence = self._extract_evidence(attn_weights, transcript)
        
        # 4. Generate Report
        print(f"\n📊 AI Risk Score: {risk_score:.4f}")
        report = self._generate_llm_report(patient_id, risk_score, evidence)
        return report

    def _load_data(self, patient_id):
        # A. Load Fused Features
        # Filename pattern: "303_fused.pt"
        feat_path = os.path.join(FEATURE_DIR, f"{patient_id}_fused.pt")
        if not os.path.exists(feat_path):
             # Try fallback pattern if exists (e.g. 303_P_fused.pt)
            feat_path = os.path.join(FEATURE_DIR, f"{patient_id}_P_fused.pt")
            
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"❌ Feature file missing: {feat_path}")
        
        features = torch.load(feat_path, map_location='cpu')
        
        # Handle cases where features are saved as a dictionary (common in this project)
        if isinstance(features, dict):
            if 'fused_embedding' in features:
                features = features['fused_embedding']
            elif 'features' in features:
                features = features['features']
            
        # Ensure it has a batch dimension [1, T, D]
        if hasattr(features, 'shape') and len(features.shape) == 2:
            features = features.unsqueeze(0)
            
        # B. Load Transcript (Rest of the method remains the same)
        trans_path = os.path.join(TRANSCRIPT_DIR, patient_id, "transcript.json")
        if not os.path.exists(trans_path):
             # Try masking folder fallback if original missing
             trans_path = os.path.join(ROOT_DIR, "phase1_rerun_outputs_masked", f"session_{patient_id}", "transcript.json")
             
        if not os.path.exists(trans_path):
            print(f"⚠️ Transcript not found at {trans_path}. Using dummy transcript.")
            transcript = []
        else:
            with open(trans_path, 'r') as f:
                data = json.load(f)
                # Handle cases where segments are in a sub-key (e.g. Whisper output)
                if isinstance(data, dict) and 'segments' in data:
                    transcript = data['segments']
                else:
                    transcript = data
                
        return features, transcript

    def _extract_evidence(self, attn_weights, transcript):
        """Finds top 3 moments where model paid attention."""
        attn = attn_weights.squeeze().cpu().numpy() # [Time]
        
        # Get Top 3 indices
        top_indices = attn.argsort()[-3:][::-1]
        
        evidence_list = []
        for idx in top_indices:
            # Map Index to Time (Window=8s, Stride=2s approx)
            timestamp = idx * 2.0
            
            # Find text snippet
            text = self._find_text_at_time(transcript, timestamp)
            
            evidence_list.append({
                "time": f"{int(timestamp//60)}:{int(timestamp%60):02d}",
                "score": float(attn[idx]),
                "text": text
            })
        return evidence_list

    def _find_text_at_time(self, transcript, timestamp, window=8.0):
        # transcript is a list of segments
        # Structure assumption: {'start': 10.0, 'end': 12.0, 'text': '...'} OR {'start_time':..., 'value':...}
        matches = []
        for seg in transcript:
            # Normalize keys
            start = seg.get('start', seg.get('start_time', 0))
            text = seg.get('text', seg.get('value', ''))
            
            if start >= timestamp and start < (timestamp + window):
                matches.append(text)
        
        return " ".join(matches) if matches else "[Non-verbal/Silence]"

    def _generate_llm_report(self, patient_id, score, evidence):
        if not hasattr(self, 'llm'):
            return "❌ Gemini not configured. Check API Key."

        risk_label = "HIGH" if score >= THRESHOLD else "LOW"
        
        prompt = f"""
        ACT AS: Expert Clinical Psychiatrist.
        TASK: Write a "Depression Screening Summary" for Patient {patient_id}.
        
        === AUTOMATED SCREENING DATA ===
        - Deep Learning Risk Score: {score:.4f} (Threshold {THRESHOLD})
        - Classification: **{risk_label} RISK**
        
        === AI EVIDENCE ANALYSIS ===
        The acoustic model flagged these moments as highly suspicious (High Attention):
        """
        
        for ev in evidence:
            prompt += f"""
            • Time {ev['time']} (Attention: {ev['score']:.2f}):
              Patient said: "{ev['text']}"
              (Context: Acoustic model detected significant distress cues here).
            """
            
        prompt += """
        === OUTPUT FORMAT ===
        Write a professional 150-word medical note.
        1. **Assessment:** State the risk level.
        2. **Observation:** Analyze the evidence. If the patient words are neutral but Attention is high, mention "Masked Depression" or "Affective Flattening".
        3. **Plan:** Recommend clinical interview or monitoring.
        """
        
        print("🤖 Consultating Gemini Pro...")
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=str, required=True, help="e.g., 303 or 302_P")
    args = parser.parse_args()

    agent = ClinicalAgent()
    try:
        report = agent.analyze_patient(args.patient_id)
        print("\n" + "="*60)
        print("📄 FINAL CLINICAL REPORT")
        print("="*60)
        print(report)
        print("="*60)
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
