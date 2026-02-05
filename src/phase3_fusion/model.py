import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    """
    Cross-Modal Attention Network for Audio-Text Fusion.
    - Audio Adapter: Projects 1024 -> d_model
    - Text Adapter: Projects 768 -> d_model
    - Cross-Attention: Text (Q) attends to Audio (K, V)
    """
    def __init__(self, 
                 d_audio=1024, 
                 d_text=768, 
                 d_model=256, 
                 nhead=8, 
                 num_layers=2,
                 dropout=0.5):
        super(FusionModel, self).__init__()
        
        # 1. Adapters (Projecting to common d_model)
        # Audio: Using a simple linear for now, could be LSTM/Transformer
        self.audio_adapter = nn.Linear(d_audio, d_model)
        self.text_adapter = nn.Linear(d_text, d_model)
        
        # 2. Cross-Modal Attention
        # Query: Text, Key/Value: Audio
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=d_model, 
                                                      num_heads=nhead, 
                                                      dropout=dropout, 
                                                      batch_first=True)
        
        # 3. Transformer Encoder Layers for further processing of fused features
        # (Optional but recommended for deeper fusion)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Binary Classification
        )
        
    def forward(self, audio_feat, text_feat, audio_mask=None, text_mask=None):
        """
        Args:
            audio_feat: [B, T_a, 1024]
            text_feat: [B, T_t, 768]
            audio_mask: [B, T_a] (1 for valid, 0 for padding) -> Needs inversion for PyTorch
            text_mask: [B, T_t] (1 for valid, 0 for padding)
        """
        # Ensure correct dtypes (robust against FP16 features)
        audio_feat = audio_feat.to(self.audio_adapter.weight.dtype)
        text_feat = text_feat.to(self.text_adapter.weight.dtype)

        # Linear projections
        audio_proj = self.audio_adapter(audio_feat) # [B, T_a, d_model]
        text_proj = self.text_adapter(text_feat)   # [B, T_t, d_model]
        
        # PyTorch MultiheadAttention uses key_padding_mask where True means MASKED
        # Our masks are 1 for valid, 0 for padding.
        if audio_mask is not None:
            key_padding_mask = ~audio_mask # [B, T_a]
        else:
            key_padding_mask = None
            
        # Cross-Attention: Text attends to Audio
        # Q = Text, K = Audio, V = Audio
        # Output is same shape as Q: [B, T_t, d_model]
        # attn_weights shape: [B, T_t, T_a] (avg over heads) or [B*NumHeads, T_t, T_a]
        attn_output, attn_weights = self.cross_modal_attn(query=text_proj, 
                                                          key=audio_proj, 
                                                          value=audio_proj, 
                                                          key_padding_mask=key_padding_mask)
        
        # Further processing with Transformer Encoder (Self-Attention on fused features)
        # Note: We can use text_mask here for self-attention
        if text_mask is not None:
            src_key_padding_mask = ~text_mask
        else:
            src_key_padding_mask = None
            
        fused_out = self.transformer_encoder(attn_output, src_key_padding_mask=src_key_padding_mask)
        
        # Global Pooling (Mean pooling over text length)
        # We only mean pool over non-padded elements
        if text_mask is not None:
            # Masked sum / count
            mask_expanded = text_mask.unsqueeze(-1).float() # [B, T_t, 1]
            pooled = (fused_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = fused_out.mean(dim=1)
            
        # Final logits
        logits = self.classifier(pooled) # [B, 1]
        
        return logits, attn_weights
