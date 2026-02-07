import os
import torch
import argparse
from tqdm import tqdm
from glob import glob
from ..phase3_fusion_v11.models import GatedMultimodalFusionV11

def extract_session_features(args):
    """
    Bridge between Phase 3.5 and Phase 4.
    Runs inference on Fusion v11 to extract session-level sequences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Phase 3.5 Model
    model = GatedMultimodalFusionV11(audio_dim=1024, text_dim=768, shared_dim=256).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logic to find all session data in data/phase2_clean/
    session_dirs = glob(os.path.join(args.data_root, "data/phase2_clean/*"))
    
    print(f"🌉 Extracting session features for {len(session_dirs)} sessions...")
    
    with torch.no_grad():
        for session_path in tqdm(session_dirs):
            p_id = os.path.basename(session_path)
            
            # Load window-level embeddings
            # Assuming files like {pid}_audio_wavlm.pt and {pid}_text_roberta.pt
            audio_path = os.path.join(session_path, f"{p_id}_audio_wavlm.pt")
            text_path = os.path.join(session_path, f"{p_id}_text_roberta.pt")
            
            if not (os.path.exists(audio_path) and os.path.exists(text_path)):
                continue
                
            audio_feat = torch.load(audio_path, map_location=device) # [T, 1024]
            text_feat = torch.load(text_path, map_location=device)   # [T, 768]
            
            # Run Fusion
            # Some sessions might have slight shape mismatches due to windowing; clip to min length
            min_len = min(audio_feat.shape[0], text_feat.shape[0])
            audio_feat = audio_feat[:min_len]
            text_feat = text_feat[:min_len]
            
            fused, gate_vals = model(audio_feat, text_feat) # [T, 256], [T, 1]
            
            # Save for Phase 4
            torch.save({
                'fused_embedding': fused.cpu(),
                'gate_values': gate_vals.cpu(),
                'patient_id': p_id
            }, os.path.join(args.output_dir, f"{p_id}_fused.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    extract_session_features(args)
