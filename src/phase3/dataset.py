import os
import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging

class MultimodalDAICDataset(Dataset):
    """
    Multimodal Dataset for E-DAIC.
    Loads paired Audio and Text embeddings with Surgical Masking for interviewer leakage.
    """
    def __init__(self, 
                 audio_dir, 
                 text_dir, 
                 labels_csv, 
                 leakage_json=None,
                 subset='train',
                 max_audio_len=5000,
                 max_text_len=1000):
        """
        Args:
            audio_dir (str): Path to audio features ({ID}_features_norm.pt).
            text_dir (str): Path to text features ({ID}_text_seq.pt).
            labels_csv (str): Path to detailed_lables.csv.
            leakage_json (str): Path to ellie_leakage_audit.json.
            subset (str): 'train', 'dev', or 'test'.
            max_audio_len (int): Max sequence length for audio.
            max_text_len (int): Max sequence length for text.
        """
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        # Load Labels
        self.df = pd.read_csv(labels_csv)
        
        # Load Leakage Audit
        self.leakage_map = {}
        if leakage_json and os.path.exists(leakage_json):
            try:
                with open(leakage_json, 'r') as f:
                    audit_data = json.load(f)
                    for entry in audit_data:
                        # Normalize ID format (session_300 -> 300)
                        sid = entry['session_id'].split('_')[-1]
                        self.leakage_map[int(sid)] = entry['leaks']
                print(f"Loaded leakage audit for {len(self.leakage_map)} sessions.")
            except Exception as e:
                print(f"Warning: Failed to load leakage audit: {e}")

        # Mapping Participant to participant_id if needed
        if 'Participant' in self.df.columns:
            self.df = self.df.rename(columns={'Participant': 'participant_id', 'Depression_label': 'label'})
            
        self.samples = []
        missing_count = 0
        
        subset_df = self.df[self.df['split'] == subset]
        
        for _, row in subset_df.iterrows():
            pid = int(row['participant_id'])
            label = int(row['label'])
            
            audio_path = os.path.join(audio_dir, f"{pid}_features.pt")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(audio_dir, f"{pid}_features_norm.pt")
            text_path = os.path.join(text_dir, f"{pid}_features.pt")
            
            if os.path.exists(audio_path) and os.path.exists(text_path):
                self.samples.append({
                    'participant_id': pid,
                    'audio_path': audio_path,
                    'text_path': text_path,
                    'label': label
                })
            else:
                missing_count += 1
                
        print(f"Loaded {len(self.samples)} samples for {subset}. Missing: {missing_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample['participant_id']
        
        # Load tensors
        audio_feat = torch.load(sample['audio_path'], map_location='cpu') # [T, 1024]
        text_feat = torch.load(sample['text_path'], map_location='cpu')   # [T, 768]
        
        # Ensure 2D: [T, D]
        if audio_feat.ndim == 3:
            audio_feat = audio_feat.view(-1, audio_feat.shape[-1])
        elif audio_feat.ndim == 1:
            audio_feat = audio_feat.unsqueeze(0)
            
        if text_feat.ndim == 3:
            text_feat = text_feat.view(-1, text_feat.shape[-1])
        elif text_feat.ndim == 1:
            text_feat = text_feat.unsqueeze(0)
            
        T = audio_feat.shape[0]
        
        # 1. Generate Surgical Mask (1 = Keep, 0 = Mask)
        surgical_mask = torch.ones(T, dtype=torch.bool)
        if pid in self.leakage_map:
            leaks = self.leakage_map[pid]
            for leak in leaks:
                l_start, l_end = leak['start'], leak['end']
                # Window i is [8i, 8(i+1)]
                # Mark any window that overlaps with the leak
                start_idx = max(0, int(l_start // 8))
                end_idx = min(T - 1, int(l_end // 8))
                surgical_mask[start_idx : end_idx + 1] = 0
        
        # Truncate
        if audio_feat.shape[0] > self.max_audio_len:
            audio_feat = audio_feat[:self.max_audio_len]
            surgical_mask = surgical_mask[:self.max_audio_len]
        if text_feat.shape[0] > self.max_text_len:
            text_feat = text_feat[:self.max_text_len]
            # Mirror text truncation if it's shorter than audio for some reason
            # (In E-DAIC T is usually equal, but we handle it)
            
        return {
            'audio_feat': audio_feat,
            'text_feat': text_feat,
            'surgical_mask': surgical_mask,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'participant_id': pid
        }

def multimodal_collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    Pads Audio and Text features and creates attention masks.
    Incorporate surgical masking (leakage filter) into the attention masks.
    """
    audio_feats = [item['audio_feat'] for item in batch]
    text_feats = [item['text_feat'] for item in batch]
    surgical_masks = [item['surgical_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    pids = [item['participant_id'] for item in batch]
    
    # Pad Audio: [B, T_a_max, 1024]
    audio_padded = pad_sequence(audio_feats, batch_first=True, padding_value=0.0)
    
    # Pad Text: [B, T_t_max, 768]
    text_padded = pad_sequence(text_feats, batch_first=True, padding_value=0.0)
    
    # Create masks (1 for real/kept data, 0 for padding/leaks)
    # Audio Mask: [B, T_a_max]
    audio_mask = torch.zeros(audio_padded.shape[0], audio_padded.shape[1], dtype=torch.bool)
    for i, s_mask in enumerate(surgical_masks):
        # Only use as much as the padded audio length allows (truncation safety)
        len_to_mask = min(s_mask.shape[0], audio_padded.shape[1])
        audio_mask[i, :len_to_mask] = s_mask[:len_to_mask]
        
    # Text Mask: [B, T_t_max]
    text_mask = torch.zeros(text_padded.shape[0], text_padded.shape[1], dtype=torch.bool)
    for i, s_mask in enumerate(surgical_masks):
        # Only use as much as the padded text length allows
        len_to_mask = min(s_mask.shape[0], text_padded.shape[1])
        text_mask[i, :len_to_mask] = s_mask[:len_to_mask]
        
    return {
        'audio_feat': audio_padded,
        'text_feat': text_padded,
        'audio_mask': audio_mask,
        'text_mask': text_mask,
        'label': labels,
        'participant_ids': pids
    }
