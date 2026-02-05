import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel
import logging

def evaluate():
    parser = argparse.ArgumentParser(description="Phase 3: Deep Evaluation")
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/03_features/audio_embeddings_clean/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/03_features/text_embeddings_seq/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_fusion_model.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--max_audio_len", type=int, default=5000)
    parser.add_argument("--max_text_len", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=256) # Must match training config
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset & Loader (Dev Set Only)
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, subset='dev', max_audio_len=args.max_audio_len, max_text_len=args.max_text_len)
    # Batch size 1 is often better for detailed analysis, but we can do bigger. 
    # For attention map, we want to grab one example.
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=multimodal_collate_fn)

    # Model
    model = FusionModel(d_model=args.d_model).to(device)
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    
    # For Attention Map (Store one interesting example, e.g., a depressed case)
    saved_attn = None
    saved_text = None # If we had text tokens, we could align them. We only have embeddings here.
    
    logger.info("Running Inference...")
    with torch.no_grad():
        for batch in dev_loader:
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, attn_weights = model(audio, text, audio_mask, text_mask)
            
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            # Save attention weights from the first batch that has a depressed sample (label=1)
            # attn_weights: [B, T_t, T_a]
            if saved_attn is None and (labels == 1).any():
                idx_in_batch = (labels == 1).nonzero(as_tuple=True)[0][0]
                saved_attn = attn_weights[idx_in_batch].cpu().numpy() # [T_t, T_a]
                # We should mask out padding for visualization
                # Get valid lengths
                t_mask = text_mask[idx_in_batch].cpu().numpy()
                a_mask = audio_mask[idx_in_batch].cpu().numpy()
                
                # Truncate to valid portion
                valid_t = t_mask.sum()
                valid_a = a_mask.sum()
                saved_attn = saved_attn[:valid_t, :valid_a]
                
    # Metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    report = classification_report(all_labels, all_preds, target_names=['Healthy', 'Depressed'])
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"F1 Score (Depressed): {f1:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Depressed'], yticklabels=['Healthy', 'Depressed'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot Attention Map
    if saved_attn is not None:
        plt.figure(figsize=(10, 8))
        # Downsample if too large for plot (e.g., audio is huge)
        # 5000 frames is too wide. Let's pool or slice for viz.
        # Let's show first 50 text tokens x first 200 audio frames
        t_limit = min(50, saved_attn.shape[0])
        a_limit = min(200, saved_attn.shape[1])
        
        subset = saved_attn[:t_limit, :a_limit]
        
        sns.heatmap(subset, cmap='viridis', cbar=True)
        plt.title('Cross-Modal Attention (Text Query x Audio Key)\nTop-Left Region')
        plt.xlabel('Audio Frames (Time)')
        plt.ylabel('Text Segments (Time)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'attention_map.png'))
        plt.close()
        logger.info(f"Attention map saved for a depressed sample.")
    else:
        logger.info("No depressed samples found in Dev set for visualization?")

    logger.info(f"Evaluation Complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    evaluate()
