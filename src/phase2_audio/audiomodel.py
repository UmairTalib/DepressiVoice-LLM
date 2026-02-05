
import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig

class DepressionAudioModel(nn.Module):
    """
    WavLM-based model for Depression Detection.
    Uses 'microsoft/wavlm-base-plus' as the backbone.
    """
    def __init__(self, 
                 model_name="microsoft/wavlm-base-plus", 
                 num_classes=2, 
                 freeze_feature_extractor=True,
                 freeze_layers_until=6):
        """
        Args:
            model_name (str): HuggingFace model hub path.
            num_classes (int): Number of output classes (2 for Binary).
            freeze_feature_extractor (bool): If True, freeze the CNN feature extractor.
            freeze_layers_until (int): Layer index. Layers 0 to this index will be frozen.
        """
        super().__init__()
        
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # Configuration for freezing
        if freeze_feature_extractor:
            self.wavlm.feature_extractor.requires_grad_(False)
            
        if freeze_layers_until > 0:
            for i in range(freeze_layers_until):
                if i < len(self.wavlm.encoder.layers):
                    for param in self.wavlm.encoder.layers[i].parameters():
                        param.requires_grad = False
                        
        hidden_size = self.wavlm.config.hidden_size # 768 for Base
        
        # Classification Head
        # Simple Mean Pooling + MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: (Batch, Time) raw audio waveform
            attention_mask: (Batch, Time) 1 for valid, 0 for padding
        """
        # WavLM Forward
        outputs = self.wavlm(input_values, attention_mask=attention_mask, output_hidden_states=True)
        
        # User requested: Select Layers 6 through 12 and compute mean
        # hidden_states is tuple of (embeddings, L1, ..., L12) -> 13 items total
        # Indices 6:13 gives layers 6, 7, 8, 9, 10, 11, 12
        all_hidden_states = outputs.hidden_states
        selected_layers = torch.stack(all_hidden_states[6:13], dim=0) # (7, Batch, SeqLen, 768)
        
        # Mean over layers
        # (Batch, SeqLen, 768)
        layer_mean = torch.mean(selected_layers, dim=0)
        batch_size, seq_len, hidden_size = layer_mean.shape
        
        # Masked Mean Pooling over Time
        if attention_mask is not None:
             # Downsample attention_mask to match seq_len
             if attention_mask.shape[1] != seq_len:
                 mask_downsampled = torch.nn.functional.interpolate(
                     attention_mask.unsqueeze(1).float(), 
                     size=seq_len, 
                     mode='nearest'
                 ).squeeze(1)
             else:
                 mask_downsampled = attention_mask
             
             mask_expanded = mask_downsampled.unsqueeze(-1)
             # Zero out padding
             layer_mean = layer_mean * mask_expanded
             
             # sum / valid_count
             sum_embeddings = torch.sum(layer_mean, dim=1)
             sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
             mean_embeddings = sum_embeddings / sum_mask
        else:
             mean_embeddings = torch.mean(layer_mean, dim=1)
             
        # Classification
        logits = self.classifier(mean_embeddings)
        
        return logits
