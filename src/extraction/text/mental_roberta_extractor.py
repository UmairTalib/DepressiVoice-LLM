import os
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_embeddings(
    input_root="data/processed/phase1/session_outputs/",
    output_dir="data/03_features/text_embeddings/",
    model_name="mental/mental-roberta-base",
    pooling_method="mean", # Options: 'mean' or 'cls'
    mode="pooled", # Options: 'pooled' (single vec) or 'sequential' (per segment)
    sliding_window=True,
    device=None,
    token=None
):
    """
    Extracts MentalRoBERTa embeddings with full context support.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info(f"Using HF token: {token[:4]}...{token[-4:]} (length: {len(token)})")
    else:
        logger.info("No HF token provided.")
        
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModel.from_pretrained(model_name, token=token).to(device)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}. It might be gated or missing: {e}")
        return

    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    session_dirs = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    logger.info(f"Found {len(session_dirs)} session directories. Mode: {mode}")
    
    success_count = 0
    fail_count = 0
    
    for session_id in tqdm(session_dirs, desc=f"Extracting ({mode})"):
        json_path = os.path.join(input_root, session_id, "cleaned_patient_text_v2.json")
        output_name = f"{session_id}_text_seq.pt" if mode == "sequential" else f"{session_id}_text.pt"
        output_path = os.path.join(output_dir, output_name)
        
        if not os.path.exists(json_path):
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if mode == "sequential":
                # Process each segment individually
                segments = data.get("segments", [])
                if not segments:
                    logger.warning(f"No segments for session {session_id}")
                    fail_count += 1
                    continue
                
                segment_embeddings = []
                for seg in segments:
                    txt = seg.get("text", "").strip()
                    if not txt:
                        # Empty segment = zero vector to maintain sequence length
                        segment_embeddings.append(torch.zeros((1, 768), dtype=torch.float16))
                        continue
                        
                    inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        emb = mean_pooling(outputs, inputs['attention_mask']) if pooling_method == "mean" else outputs.last_hidden_state[:, 0, :]
                        
                        # Normalize
                        mean = emb.mean(dim=-1, keepdim=True)
                        std = emb.std(dim=-1, keepdim=True)
                        norm_emb = (emb - mean) / (std + 1e-5)
                        segment_embeddings.append(norm_emb.to(torch.float16).cpu())
                
                final_tensor = torch.cat(segment_embeddings, dim=0) # [Num_Segments, 768]
                torch.save(final_tensor, output_path)
                success_count += 1

            else:
                # Pooled Mode (Sliding Window for Full Context)
                text = data.get("text", "").strip()
                if not text:
                    logger.warning(f"Empty transcript for session {session_id}")
                    fail_count += 1
                    continue
                
                # Full tokenization
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
                input_ids = inputs['input_ids'][0]
                total_tokens = len(input_ids)
                
                if sliding_window and total_tokens > 512:
                    chunk_size = 512
                    stride = 462 # 50 token overlap
                    chunks = []
                    
                    for i in range(0, total_tokens, stride):
                        chunk_ids = input_ids[i:i+chunk_size].unsqueeze(0).to(device)
                        if chunk_ids.size(1) < 5: continue # Skip tiny fragments
                        
                        with torch.no_grad():
                            # Re-add attention mask for the chunk
                            mask = torch.ones_like(chunk_ids).to(device)
                            outputs = model(chunk_ids, attention_mask=mask)
                            # Create a dummy inputs dict for mean_pooling
                            dummy_inputs = {'attention_mask': mask}
                            emb = mean_pooling(outputs, dummy_inputs['attention_mask']) if pooling_method == "mean" else outputs.last_hidden_state[:, 0, :]
                            chunks.append(emb)
                    
                    # Average all chunk embeddings
                    embedding = torch.stack(chunks).mean(dim=0)
                else:
                    # Standard execution
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = mean_pooling(outputs, inputs['attention_mask']) if pooling_method == "mean" else outputs.last_hidden_state[:, 0, :]
                
                # Final Normalization
                mean = embedding.mean(dim=-1, keepdim=True)
                std = embedding.std(dim=-1, keepdim=True)
                normalized_embedding = (embedding - mean) / (std + 1e-5)
                
                if torch.isnan(normalized_embedding).any():
                    fail_count += 1
                    continue
                    
                torch.save(normalized_embedding.to(torch.float16).cpu(), output_path)
                success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process session {session_id}: {e}")
            fail_count += 1
            
    logger.info(f"Extraction complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MentalRoBERTa embeddings.")
    parser.add_argument("--input_root", type=str, default="data/processed/phase1/session_outputs/", help="Root directory for session outputs.")
    parser.add_argument("--output_dir", type=str, default="data/03_features/text_embeddings/", help="Directory to save embeddings.")
    parser.add_argument("--model_name", type=str, default="mental/mental-roberta-base", help="HuggingFace model name.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling method.")
    parser.add_argument("--mode", type=str, default="pooled", choices=["pooled", "sequential"], help="Extraction mode.")
    parser.add_argument("--sliding_window", action="store_true", default=True, help="Use sliding window for pooled mode.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token.")
    
    args = parser.parse_args()
    
    extract_embeddings(
        input_root=args.input_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pooling_method=args.pooling,
        mode=args.mode,
        sliding_window=args.sliding_window,
        device=args.device,
        token=args.hf_token
    )
