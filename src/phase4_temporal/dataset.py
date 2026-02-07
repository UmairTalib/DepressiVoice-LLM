import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, data_dir, labels_csv, augment=False):
        """
        Args:
            data_dir: Directory containing {id}_fused.pt files
            labels_csv: Path to master_labels_v7.csv
            augment: If True, applies slicing augmentation
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.patients = self.labels_df['Participant'].astype(str).tolist()
        self.augment = augment
        
        # Pre-filter patients that actually have feature files
        self.available_patients = []
        for p in self.patients:
            if os.path.exists(os.path.join(data_dir, f"{p}_fused.pt")):
                self.available_patients.append(p)
            elif os.path.exists(os.path.join(data_dir, f"{p}_P_fused.pt")):
                self.available_patients.append(f"{p}_P")

    def __len__(self):
        return len(self.available_patients)

    def __getitem__(self, idx):
        p_id = self.available_patients[idx]
        actual_pid = p_id.replace("_P", "")
        
        # Load features
        feat_path = os.path.join(self.data_dir, f"{p_id}_fused.pt")
        data = torch.load(feat_path, map_location='cpu')
        
        # MTL Support: Get binary label and PHQ score
        row = self.labels_df[self.labels_df['Participant'].astype(str) == actual_pid].iloc[0]
        label = int(row['PHQ_Binary'])
        phq_score = float(row['PHQ_Score_Norm']) # Normalized 0-1
        
        features = data['fused_embedding'] # [T, 256]
        
        if self.augment:
            # Slicing augmentation (v4): Take first or second half
            mid = features.shape[0] // 2
            if np.random.rand() > 0.5:
                features = features[:mid]
            else:
                features = features[mid:]
        
        return {
            'features': features,
            'label': label,
            'phq_score': phq_score,
            'patient_id': p_id
        }

def pad_collate(batch):
    features = [b['features'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    phq_scores = torch.tensor([b['phq_score'] for b in batch], dtype=torch.float32)
    
    # Pad sequences: [Batch, Max_T, 256]
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    # Create mask for Transformer (True where padding exists)
    mask = torch.zeros(padded_features.shape[:2], dtype=torch.bool)
    for i, feat in enumerate(features):
        mask[i, len(feat):] = True
        
    return {
        'features': padded_features,
        'mask': mask,
        'labels': labels,
        'phq_scores': phq_scores
    }
