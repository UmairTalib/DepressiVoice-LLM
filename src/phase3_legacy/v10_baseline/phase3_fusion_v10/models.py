import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionV10Model(nn.Module):
    def __init__(self, 
                 audio_dim: int = 1024,
                 text_dim: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.35):
        super(FusionV10Model, self).__init__()
        
        # Audio Encoder: LSTM
        # Input: [Batch, T, 1024]
        # Output: [Batch, T, Hidden] (if batch_first=True)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Text Encoder: LSTM
        # Input: [Batch, T, 768]
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.text_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Asymmetric Cross-Attention Fusion (Text queries Audio)
        # Q = Text, K = Audio, V = Audio
        # key_padding_mask will be used here to ignore zeroed audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classification Head
        # Input dim = hidden_dim (from fusion) + 3 (scalars)
        head_input_dim = hidden_dim + 3
        
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Binary Logit
        )

    def forward(self, audio, text, scalars):
        """
        audio: [Batch, T, 1024]
        text: [Batch, T, 768]
        scalars: [Batch, 3]
        """
        
        # 1. Generate Mask for Audio
        # Detect simple zero vectors (assuming perfect zeros from sanitization)
        # shape: [Batch, T]
        # True = Ignored/Padding
        audio_abs_sum = audio.abs().sum(dim=-1)
        key_padding_mask = (audio_abs_sum == 0) 
        
        # Prevent all-masked sequence which causes NaNs in Softmax
        # If all steps are masked, unmask the first one (it will be zero input, but prevents NaN)
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask[all_masked, 0] = False
        
        # 2. Encode Audio
        # [B, T, 1024] -> [B, T, H]
        a_emb = self.audio_proj(audio)
        # LSTM processes everything, including zeros. 
        # State will drift, but Attention will ignore the output at zero steps.
        a_enc, _ = self.audio_lstm(a_emb)
        
        # 3. Encode Text
        # [B, T, 768] -> [B, T, H]
        t_emb = self.text_proj(text)
        t_enc, _ = self.text_lstm(t_emb)
        
        # 4. Fusion: Text queries Audio
        # Q=Text, K=Audio, V=Audio
        # Pass key_padding_mask to ignore zeroed audio steps in Attention
        fused, _ = self.cross_attn(
            query=t_enc, 
            key=a_enc, 
            value=a_enc, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual + Norm
        fused = self.norm(fused + t_enc)
        
        # 5. Pooling
        # [B, T, H] -> [B, H]
        # We should NOT pool over the masked steps for the Query side either?
        # Actually, Query is Text. Text might be silent too?
        # If Text is silent/masked, we probably shouldn't use it for final decision.
        # But 'text' wasn't strictly zeroed in dataset.py, only implies mask.
        # Let's use simple mean pooling for now, or attention-masked pooling.
        # Mean Pooling over time
        pooled = torch.mean(fused, dim=1)
        
        # 6. Injection
        # Concat scalars
        combined = torch.cat([pooled, scalars], dim=1)
        
        # 7. Head
        logits = self.head(combined)
        
        return logits
