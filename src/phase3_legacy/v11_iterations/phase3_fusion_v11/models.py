import torch
import torch.nn as nn

class CoAttentionFusion(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=768, shared_dim=256, nhead=8, dropout=0.4):
        super(CoAttentionFusion, self).__init__()
        
        # 1. Dimension Projection
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        
        self.norm_a = nn.LayerNorm(shared_dim)
        self.norm_t = nn.LayerNorm(shared_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Parallel Co-Attention Blocks
        self.audio_query_attn = nn.MultiheadAttention(shared_dim, nhead, dropout=dropout, batch_first=True)
        self.text_query_attn = nn.MultiheadAttention(shared_dim, nhead, dropout=dropout, batch_first=True)
        
        # 3. Gated Multimodal Unit (GMU)
        self.fusion_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid()
        )

    def forward(self, audio, text, audio_mask=None, text_mask=None):
        """
        audio: [Batch, T, audio_dim]
        text: [Batch, T, text_dim]
        audio_mask: [Batch, T] (True = Padding/Ignored) - Optional
        text_mask: [Batch, T] (True = Padding/Ignored) - Optional
        """
        # Strict Output Masking
        if text_mask is not None:
            # Safeguard: Prevent all-True masks which cause NaN in Attention
            # Logic: If a modality is fully masked, we unmask the first token (which is zero) 
            # so Attention has *something* to attend to (weight=1.0 on zero), resulting in zero output.
            
            # Check text_mask (used as Key for Audio Query)
            if float(torch.__version__[:3]) >= 2.0: # Optimizations in 2.0+ might handle this, but explicit safe is better
                 text_all_masked = text_mask.all(dim=1)
                 if text_all_masked.any():
                     text_mask = text_mask.clone()
                     text_mask[text_all_masked, 0] = False
            
            # Check audio_mask (used as Key for Text Query)
            if audio_mask is not None:
                audio_all_masked = audio_mask.all(dim=1)
                if audio_all_masked.any():
                    audio_mask = audio_mask.clone()
                    audio_mask[audio_all_masked, 0] = False
        
        # --- Projections ---
        a_proj = self.dropout(self.norm_a(self.audio_proj(audio)))
        t_proj = self.dropout(self.norm_t(self.text_proj(text)))
        
        # --- Bidirectional Co-Attention ---
        # Query=Audio, Key=Text, Value=Text
        # key_padding_mask should mask Keys (Text)
        head_a, _ = self.audio_query_attn(query=a_proj, key=t_proj, value=t_proj, key_padding_mask=text_mask)
        
        # Query=Text, Key=Audio, Value=Audio
        # key_padding_mask should mask Keys (Audio)
        head_b, _ = self.text_query_attn(query=t_proj, key=a_proj, value=a_proj, key_padding_mask=audio_mask)
        
        # --- Gated Fusion (GMU) ---
        gate_input = torch.cat([head_a, head_b], dim=-1)
        z = self.fusion_gate(gate_input) # Learned Gate Value
        
        fused_output = z * head_a + (1 - z) * head_b
        
        # Residual Connections for stability
        fused_output = fused_output + a_proj + t_proj
        
        # Strict Output Masking
        if text_mask is not None:
            out_mask = text_mask.unsqueeze(-1).logical_not().float() 
            fused_output = fused_output * out_mask
            
        return fused_output, z # Return z for explainability/plotting

class FusionV11Classifier(nn.Module):
    """
    Wrapper around CoAttentionFusion to add a classification head for Phase 3 training.
    """
    def __init__(self, 
                 audio_dim=1024, 
                 text_dim=768, 
                 shared_dim=256, 
                 nhead=8, 
                 dropout=0.4,
                 num_scalars=3):
        super(FusionV11Classifier, self).__init__()
        
        self.fusion = CoAttentionFusion(audio_dim, text_dim, shared_dim, nhead, dropout)
        
        # Classification Head
        # Input: shared_dim (pooled) + num_scalars
        head_input_dim = shared_dim + num_scalars
        
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Binary Logit
        )

    def forward(self, audio, text, scalars, audio_mask=None, text_mask=None):
        """
        audio: [Batch, T, 1024]
        text: [Batch, T, 768]
        scalars: [Batch, 3]
        """
        
        # Automatically generate masks if not provided
        # Assuming Data is padded with 0s? 
        # Or just rely on what Dataset provides passed in as args. (Currently dataset.py doesn't return masks explicitly in dict, need to add logic)
        # Let's infer mask from data if None
        if text_mask is None:
             # Heuristic: If text vector is all zeros, it's padding
            text_mask = (text.abs().sum(dim=-1) == 0)
            
        if audio_mask is None:
            audio_mask = (audio.abs().sum(dim=-1) == 0)
            
        # Run Fusion
        fused_seq, z = self.fusion(audio, text, audio_mask, text_mask) # [Batch, T, Shared]
        
        # Pooling
        # Mean Pool over Time (ignoring masked steps)
        # mask: True=Ignored. 
        # numeric_mask: 1=Valid, 0=Ignored
        numeric_mask = text_mask.logical_not().float().unsqueeze(-1) # [Batch, T, 1]
        
        # Sum valid steps
        sum_pooled = (fused_seq * numeric_mask).sum(dim=1) # [Batch, Shared]
        
        # Count valid steps
        count_valid = numeric_mask.sum(dim=1) # [Batch, 1]
        count_valid = count_valid.clamp(min=1.0) # Avoid div by zero
        
        pooled = sum_pooled / count_valid
        
        # Inject Scalars
        combined = torch.cat([pooled, scalars], dim=1)
        
        # Head
        logits = self.head(combined)
        
        return logits, z
