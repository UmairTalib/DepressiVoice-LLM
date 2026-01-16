"""
DepressiVoice-LLM Data Processing Module

This module handles loading audio data, transcription using Whisper,
and processing for depression detection analysis.
Optimized for CUDA acceleration on Omni cluster (Uni Siegen).
"""

import os
import yaml
import torch
# --- AGGRESSIVE MONKEY PATCH FOR PYTORCH 2.6+ ---
# Must be applied BEFORE any other library imports
try:
    _original_load = torch.load
    def _safe_load_shim(*args, **kwargs):
        # Force weights_only=False to allow loading older checkpoints
        # containing OmegaConf/List/etc. regardless of caller's intent.
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load_shim
    
    # Also whitelist OmegaConf for Pyannote
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
except Exception as e:
    print(f"WARNING: Patch failed: {e}")
# ------------------------------------------------

from typing import Optional, Dict, Any

import pandas as pd
import whisperx
import gc
import librosa
from tqdm import tqdm

# Fix for DiarizationPipeline attribute error
try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

def check_cuda_availability() -> Dict[str, Any]:
    """
    Check CUDA availability and GPU information for Omni cluster.
    
    Returns:
        Dictionary with CUDA status and GPU information.
    """
    cuda_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': '.'.join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'gpu_name': None,
        'gpu_memory': None
    }
    
    if cuda_info['cuda_available']:
        cuda_info['current_device'] = torch.cuda.current_device()
        cuda_info['gpu_name'] = torch.cuda.get_device_name(0)
        cuda_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        print("="*50)
        print("CUDA INFORMATION (Omni Cluster)")
        print("="*50)
        print(f"CUDA Available: {cuda_info['cuda_available']}")
        print(f"CUDA Version: {cuda_info['cuda_version']}")
        print(f"GPU Count: {cuda_info['gpu_count']}")
        print(f"Current Device: {cuda_info['current_device']}")
        print(f"GPU Name: {cuda_info['gpu_name']}")
        print(f"GPU Memory: {cuda_info['gpu_memory']:.2f} GB")
        print("="*50)
    else:
        print("WARNING: CUDA not available. Running on CPU.")
        
    return cuda_info


def setup_cuda_environment() -> torch.device:
    """
    Setup optimal CUDA environment for Omni cluster.
    
    Returns:
        torch.device object for computation.
    """
    if torch.cuda.is_available():
        # Set CUDA device
        device = torch.device("cuda")
        
        # Optimize for Omni cluster
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        print(f"INFO: Using CUDA device: {device}")
        return device
    else:
        device = torch.device("cpu")
        print("INFO: CUDA not available, using CPU")
        return device


def load_dataset_manifest(config_path: str = "config.yaml") -> Optional[pd.DataFrame]:
    """
    Load the main metadata file to map audio files to labels.

    Args:
        config_path: Path to the project's config file.

    Returns:
        DataFrame containing the dataset manifest, or None on error.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        data_config = config['data_processing']
        path_config = config['paths']

        manifest_filename = data_config['manifest_filename']
        manifest_path = os.path.join(path_config['raw_data'], manifest_filename)

        print(f"INFO: Loading dataset manifest from: {manifest_path}")

        if not os.path.exists(manifest_path):
            print(f"ERROR: Manifest file not found at '{manifest_path}'.")
            print("Please check the paths in your config.yaml.")
            return None

        manifest_df = pd.read_csv(manifest_path)
        print(f"INFO: Successfully loaded manifest with {len(manifest_df)} records.")
        return manifest_df

    except FileNotFoundError:
        print(f"ERROR: Config file not found at '{config_path}'.")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML format in config file: {e}")
        return None
    except KeyError as e:
        print(f"ERROR: Missing required key in config: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return None


def run_transcription(config: Dict[str, Any], manifest_df: pd.DataFrame, file_suffix: str = "") -> Optional[pd.DataFrame]:
    """
    Transcribe all audio files using Whisper with CUDA acceleration.
    Optimized for Omni cluster at Uni Siegen.

    Args:
        config: The project configuration dictionary.
        manifest_df: DataFrame with participant and audio file info.
        file_suffix: Optional suffix for output filename (e.g., '_part_0').
        
    Returns:
        DataFrame with transcripts merged with original manifest data.
    """
    # Setup CUDA environment
    cuda_info = check_cuda_availability()
    compute_type = "float16" if cuda_info['cuda_available'] else "float32"
    device_str = "cuda" if cuda_info['cuda_available'] else "cpu"
    
    model_name = config['model']['whisper_model']
    print(f"INFO: Loading WhisperX model: {model_name}")
    
    try:
        # 1. Transcribe with original whisper (acceleration via faster-whisper)
        model = whisperx.load_model(
            model_name, 
            device_str, 
            compute_type=compute_type,
            language='en'
        )
        print(f"INFO: WhisperX model loaded on {device_str}")
            
    except Exception as e:
        print(f"ERROR: Failed to load WhisperX model: {e}")
        return None

    transcripts = []
    target_sr = config['audio']['sample_rate']
    audio_format = config['audio']['format']
    hf_token = config['model'].get('hf_token')
    
    # Load Alignment model
    try:
        print("INFO: Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code="en", 
            device=device_str
        )
    except Exception as e:
        print(f"WARNING: Alignment model failed to load. Skipping alignment/diarization. Error: {e}")
        model_a = None

    # Load Diarization model
    diarize_model = None
    if config['diarization'].get('enabled', False) and hf_token:
        try:
            print("INFO: Loading diarization model...")
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token, 
                device=device_str
            )
        except Exception as e:
            print(f"WARNING: Diarization model failed to load. Check HF Token. Error: {e}")
    else:
        print("INFO: Diarization disabled or Token missing.")

    print("INFO: Starting WhisperX transcription process...")

    # Process each audio file
    for index, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Processing"):
        participant_id = row['Participant_ID']
        
        # Build audio file path
        audio_filename = f"{participant_id}_AUDIO.{audio_format}"
        audio_path = os.path.join(config['paths']['raw_data'], audio_filename)

        if not os.path.exists(audio_path):
            print(f"WARNING: Audio file not found for participant {participant_id} at {audio_path}")
            continue

        try:
            # 1. Transcribe
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=16)
            
            # 2. Align (if available)
            if model_a:
                try:
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device_str, 
                        return_char_alignments=False
                    )
                except Exception as e:
                    print(f"WARNING: Alignment failed for {participant_id}: {e}")

            # 3. Diarize (if available)
            if diarize_model:
                try:
                    diarize_segments = diarize_model(audio, min_speakers=config['diarization']['min_speakers'], max_speakers=config['diarization']['max_speakers'])
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    print(f"WARNING: Diarization failed for {participant_id}: {e}")

            # Collect segments
            if 'segments' in result and result['segments']:
                for segment in result['segments']:
                    # Filter out segments with unknown speakers if requested
                    speaker = segment.get('speaker', 'unknown')
                    if speaker == 'unknown':
                        continue

                    # Calculate confidence score robustly
                    # 1. Try average of word scores (Most accurate for WhisperX)
                    if 'words' in segment:
                        word_scores = [w.get('score', 0.0) for w in segment['words'] if 'score' in w]
                        if word_scores:
                            confidence = sum(word_scores) / len(word_scores)
                        else:
                            confidence = 0.0
                    # 2. Fallback to segment score
                    else:
                        confidence = segment.get('score', 0.0)
                    
                    # 3. Fallback to log probability (converted)
                    if confidence == 0.0 and 'avg_logprob' in segment:
                        import math
                        confidence = math.exp(segment['avg_logprob'])

                    # Skip very low confidence segments (e.g. hallucinations)
                    if confidence < 0.2:
                        continue

                    transcripts.append({
                        'Participant_ID': participant_id,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'transcript': segment['text'].strip(),
                        'speaker': speaker,
                        'confidence': round(confidence, 4),
                        'language': result.get('language', 'en') # Default to 'en' since we forced it
                    })
            
            # Cleanup garbage to save VRAM
            if cuda_info['cuda_available'] and index % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"WARNING: Could not process file for participant {participant_id}. Error: {e}")
            if cuda_info['cuda_available']:
                torch.cuda.empty_cache()

    if not transcripts:
        print("ERROR: No transcripts were generated.")
        return None

    # Create DataFrame and merge
    transcripts_df = pd.DataFrame(transcripts)
    output_df = pd.merge(manifest_df, transcripts_df, on='Participant_ID', how='right')

    # Save results
    base_name = 'transcripts_whisperx'
    output_filename = f"{base_name}{file_suffix}.csv"
    output_path = os.path.join(config['paths']['transcribed_data'], output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\nINFO: WhisperX processing complete!")
        print(f"INFO: Output saved to {output_path}")
        return output_df
        
    except Exception as e:
        print(f"ERROR: Failed to save transcripts: {e}")
        return None
    finally:
        # Final CUDA cleanup
        if cuda_info['cuda_available']:
            torch.cuda.empty_cache()


import argparse
import numpy as np

def main() -> None:
    """Main function to run the CUDA-optimized data processing pipeline."""
    parser = argparse.ArgumentParser(description="DepressiVoice-LLM Transcription")
    parser.add_argument("--shard-index", type=int, default=0, help="Index of current shard (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    config_path = "config.yaml"
    
    print("="*60)
    print(f"DepressiVoice-LLM Data Processing (Shard {args.shard_index+1}/{args.num_shards})")
    print("="*60)
    
    # Load configuration
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return
    
    # Load dataset manifest
    manifest_df = load_dataset_manifest(config_path)
    if manifest_df is None:
        print("ERROR: Failed to load dataset manifest. Exiting.")
        return
    
    # --- SHARDING LOGIC ---
    if args.num_shards > 1:
        # Split dataframe into N parts
        shards = np.array_split(manifest_df, args.num_shards)
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            print(f"ERROR: Invalid shard index {args.shard_index}. Must be 0 to {args.num_shards-1}")
            return
            
        manifest_df = shards[args.shard_index]
        print(f"INFO: Processing Shard {args.shard_index} with {len(manifest_df)} records.")
        
        # Update output path to include shard index
        # We handle this by modifying the return value of run_transcription or config
        # But simpler is to patch the config path temporarily or handle it in saving.
        # Let's handle it by passing the suffix to run_transcription.
    else:
        print("INFO: Processing entire dataset (Single Shard).")

    # Run CUDA-accelerated transcription
    # We need to modify run_transcription signature slightly or append to output path inside it?
    # Let's modify run_transcription to take an optional suffix
    result_df = run_transcription(config, manifest_df, file_suffix=f"_part_{args.shard_index}" if args.num_shards > 1 else "")
    
    if result_df is None:
        print("ERROR: Transcription failed. Exiting.")
        return
    
    print("INFO: CUDA-optimized data processing completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()