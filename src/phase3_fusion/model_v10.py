import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionV10Model(nn.Module):
    """
    Fusion V10: Asymmetric Cross-Attention with Scalar Injection.
    """
    def __init__(self, 
                 d_audio=1024, 
                 d_text=768, 
                 d_model=768, # Projected dimension (Project Audio to Text dim)
                 nhead=8, 
                 num_layers=2, # For Transformer Encoder (optional, spec says Assym Attn directly?)
                 # Spec: "Stream A: Text queries Audio", "Stream B: Audio queries Text".
                 # It implies raw attention blocks.
                 dropout_transformer=0.1,
                 dropout_head=0.35):
        super(FusionV10Model, self).__init__()
        
        # 1. Projectors
        # Text is already 768. We assume it's passed as embeddings.
        # Audio is 1024. Project to 768.
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.text_norm = nn.LayerNorm(d_model) # Stabilization
        self.audio_norm = nn.LayerNorm(d_model)
        
        # 2. Asymmetric Cross-Attention
        # Stream A: Text (Q) queries Audio (K,V) -> Main Signal
        self.attn_text_query = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_transformer, batch_first=True)
        
        # Stream B: Audio (Q) queries Text (K,V) -> Aux Signal
        self.attn_audio_query = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_transformer, batch_first=True)
        
        # 3. Scalar Injection point
        # After pooling, we have d_model vec. We concat 3 scalars.
        self.d_final = d_model + 3
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_final, 256),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(256, 1)
        )
        
    def forward(self, audio_feat, text_feat, scalars, audio_mask=None, text_mask=None):
        """
        Args:
            audio_feat: [B, T_a, 1024]
            text_feat: [B, T_t, 768]
            scalars: [B, 3] (Normalized)
            audio_mask: [B, T_a] (1 for valid)
            text_mask: [B, T_t]
        """
        # 1. Project Audio to match Text dim
        audio_emb = self.audio_norm(self.audio_proj(audio_feat)) # [B, T_a, 768]
        text_emb = self.text_norm(text_feat)                     # [B, T_t, 768]
        
        # Masks for Attention (inverted: 1 for pad)
        # PyTorch MHA expects key_padding_mask where True=Ignore
        key_padding_mask_audio = (audio_mask == 0) if audio_mask is not None else None
        key_padding_mask_text = (text_mask == 0) if text_mask is not None else None
        
        # 2. Asymmetric Attention
        
        # Stream A: Text queries Audio
        # Q=Text, K=Audio, V=Audio
        # Output shape: [B, T_t, 768] (Aligned to Text)
        stream_a, _ = self.attn_text_query(query=text_emb, 
                                           key=audio_emb, 
                                           value=audio_emb, 
                                           key_padding_mask=key_padding_mask_audio)
        
        # Stream B: Audio queries Text
        # Q=Audio, K=Text, V=Text
        # Output shape: [B, T_a, 768] (Aligned to Audio)
        stream_b, _ = self.attn_audio_query(query=audio_emb, 
                                            key=text_emb, 
                                            value=text_emb, 
                                            key_padding_mask=key_padding_mask_text)
        
        # 3. Pooling & Combination
        # We need a single vector representing the window.
        # Spec says: "Injection: Concatenate the 3 Scalars to the fused embedding"
        # Implies we need to pool Stream A and Stream B first?
        # Or do we fuse sequence-wise?
        # "Fused = Stream A + 0.5 * Stream B"
        # Issue: Stream A is lengths T_t, Stream B is lengths T_a. They cannot be added element-wise!
        
        # Interpretation of User Spec:
        # "Stream B... weighted 0.5".
        # This implies we pool them first, then add.
        # OR we perform Global Average Pooling on both, then combine.
        
        # Let's GAP (Global Average Pool) both streams.
        # Use masks to avoid pooling padding.

        # Pool Stream A (Text-aligned)
        if text_mask is not None:
             mask_t_expanded = text_mask.unsqueeze(-1)
             pool_a = (stream_a * mask_t_expanded).sum(dim=1) / (mask_t_expanded.sum(dim=1) + 1e-9)
        else:
             pool_a = stream_a.mean(dim=1)
             
        # Pool Stream B (Audio-aligned)
        if audio_mask is not None:
             mask_a_expanded = audio_mask.unsqueeze(-1)
             pool_b = (stream_b * mask_a_expanded).sum(dim=1) / (mask_a_expanded.sum(dim=1) + 1e-9)
        else:
             pool_b = stream_b.mean(dim=1)
             
        # Fused Vector
        fused = pool_a + 0.5 * pool_b
        
        # 4. Injection
        # Concatenate Scalars
        features = torch.cat([fused, scalars], dim=1) # [B, 768+3]
        
        # 5. Classifier
        logits = self.classifier(features)
        
        return logits, None # No attn weights returned for now
