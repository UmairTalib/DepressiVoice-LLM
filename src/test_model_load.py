import os
import yaml
import torch
import whisperx
import gc

# --- AGGRESSIVE MONKEY PATCH FOR PYTORCH 2.6+ ---
try:
    _original_load = torch.load
    def _safe_load_shim(*args, **kwargs):
        # Force weights_only=False to allow loading older checkpoints
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load_shim
    
    # Whitelist all required classes
    import omegaconf
    import omegaconf.base
    import typing
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        typing.Any,
        typing.List,
        typing.Dict,
        typing.Optional
    ])
    print("PATCH APPLIED SUCCESSFULLY")
except Exception as e:
    print(f"WARNING: Patch failed: {e}")
# ------------------------------------------------

def test_load():
    print("Testing PyTorch version:", torch.__version__)
    print("Testing CUDA:", torch.cuda.is_available())
    
    try:
        device = "cuda"
        compute_type = "float16"
        print("Attempting to load WhisperX model...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        print("SUCCESS: WhisperX model loaded!")
        
        # Test Diarization Loading (Models often fail here)
        print("Attempting to load Diarization pipeline...")
        # Need HF Token from config or env
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
        
        if hf_token:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            print("SUCCESS: Diarization pipeline loaded!")
        else:
            print("SKIPPING Diarization test (No Token found)")

    except Exception as e:
        print("FAILED!")
        print(e)
        raise e

if __name__ == "__main__":
    test_load()
