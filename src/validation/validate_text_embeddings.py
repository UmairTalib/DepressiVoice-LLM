import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_embeddings(
    embedding_dir="data/03_features/text_embeddings_v2/",
    labels_path="data/01_raw/detailed_lables.csv"
):
    """
    Validates the statistical integrity of extracted text embeddings.
    """
    if not os.path.exists(embedding_dir):
        logger.error(f"Embedding directory not found: {embedding_dir}")
        return

    # Load labels
    try:
        df_labels = pd.read_csv(labels_path)
        # Assuming ID and PHQ8_Binary columns
        if 'ID' not in df_labels.columns:
            # Try to fix column names if they are different
            df_labels = df_labels.rename(columns={df_labels.columns[0]: 'ID'})
        
        # Filter for binary labels if available
        label_col = 'PHQ8_Binary' if 'PHQ8_Binary' in df_labels.columns else df_labels.columns[-1]
        labels_dict = dict(zip(df_labels['ID'].astype(str), df_labels[label_col]))
    except Exception as e:
        logger.error(f"Failed to load labels: {e}")
        labels_dict = {}

    files = [f for f in os.listdir(embedding_dir) if f.endswith('.pt')]
    if not files:
        logger.error("No embeddings found to validate.")
        return

    all_embeddings = []
    session_ids = []
    
    logger.info(f"Loading {len(files)} embeddings for validation...")
    for f in tqdm(files):
        try:
            emb = torch.load(os.path.join(embedding_dir, f))
            if isinstance(emb, torch.Tensor):
                all_embeddings.append(emb.float().numpy().flatten())
                session_ids.append(f.split('_')[0])
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not all_embeddings:
        return

    X = np.array(all_embeddings)
    
    # 1. Statistical Checks
    mean = np.mean(X)
    std = np.std(X)
    var = np.var(X)
    
    # 2. Anisotropy (Cosine Similarity)
    def compute_avg_cosine_sim(X):
        # Sample 100 random pairs to avoid quadratic memory issues
        if len(X) > 100:
            idx = np.random.choice(len(X), 100, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
            
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(X_sample)
        # Average of off-diagonal elements
        avg_sim = (np.sum(sims) - len(X_sample)) / (len(X_sample) * (len(X_sample) - 1))
        return avg_sim

    avg_sim = compute_avg_cosine_sim(X)

    # 3. Variance per dimension
    dim_vars = np.var(X, axis=0)
    avg_dim_var = np.mean(dim_vars)

    # 4. Correlation with labels (if available)
    correlation_score = "N/A"
    if labels_dict:
        y = []
        X_valid = []
        for i, sid in enumerate(session_ids):
            if sid in labels_dict:
                y.append(labels_dict[sid])
                X_valid.append(X[i])
        
        if len(y) > 10:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, np.array(X_valid), np.array(y), cv=5)
            correlation_score = f"{np.mean(scores):.4f} (+/- {np.std(scores):.4f})"

    # Report
    print("\n" + "="*50)
    print("EMBEDDING VALIDATION REPORT")
    print("="*50)
    print(f"Total Sessions:      {len(X)}")
    print(f"Embedding Dim:       {X.shape[1]}")
    print(f"Global Mean:         {mean:.6f} (Target: ~0)")
    print(f"Global Std:          {std:.6f} (Target: ~1)")
    print(f"Avg Dim Variance:    {avg_dim_var:.6f}")
    print(f"CosSim (Anisotropy): {avg_sim:.6f} (Lower = Better)")
    print(f"Linear Probe Acc:    {correlation_score}")
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/03_features/text_embeddings_v2/")
    parser.add_argument("--labels", type=str, default="data/01_raw/detailed_lables.csv")
    args = parser.parse_args()
    
    validate_embeddings(args.dir, args.labels)
