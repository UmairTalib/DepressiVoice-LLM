"""
DepressiVoice-LLM Data Processing Module

This module handles loading audio data, transcription using Whisper,
and processing for depression detection analysis.
Optimized for CUDA acceleration on Omni cluster (Uni Siegen).
"""

import os
import yaml
import torch
from typing import Optional, Dict, Any

import pandas as pd
import whisper
import librosa
from tqdm import tqdm


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


def run_transcription(config: Dict[str, Any], manifest_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Transcribe all audio files using Whisper with CUDA acceleration.
    Optimized for Omni cluster at Uni Siegen.

    Args:
        config: The project configuration dictionary.
        manifest_df: DataFrame with participant and audio file info.
        
    Returns:
        DataFrame with transcripts merged with original manifest data.
    """
    # Setup CUDA environment
    cuda_info = check_cuda_availability()
    device = setup_cuda_environment()
    
    print(f"INFO: Loading Whisper model: {config['model']['whisper_model']}")
    
    try:
        # Load Whisper model with CUDA optimization
        if cuda_info['cuda_available']:
            # Use FP16 for faster inference on CUDA
            model = whisper.load_model(
                config['model']['whisper_model'],
                device=device
            )
            print(f"INFO: Whisper model loaded on {device} with FP16 optimization")
        else:
            # Fallback to CPU
            model = whisper.load_model(config['model']['whisper_model'])
            print("INFO: Whisper model loaded on CPU")
            
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model: {e}")
        return None

    transcripts = []
    target_sr = config['audio']['sample_rate']
    audio_format = config['audio']['format']
    
    print(f"INFO: Audio will be standardized to {target_sr} Hz.")
    print("INFO: Starting CUDA-accelerated transcription process...")

    # Process each audio file with progress bar
    for index, row in tqdm(
        manifest_df.iterrows(), 
        total=len(manifest_df), 
        desc="Transcribing Audio (CUDA)" if cuda_info['cuda_available'] else "Transcribing Audio (CPU)"
    ):
        participant_id = row['Participant_ID']
        
        # Build audio file path
        audio_filename = f"{participant_id}_AUDIO.{audio_format}"
        audio_path = os.path.join(config['paths']['raw_data'], audio_filename)

        if not os.path.exists(audio_path):
            print(f"WARNING: Audio file not found for participant {participant_id} at {audio_path}")
            continue

        try:
            # Load and resample audio to target sample rate
            audio_array, _ = librosa.load(audio_path, sr=target_sr, mono=True)
            
            # Transcribe with CUDA optimization
            if cuda_info['cuda_available']:
                # Use FP16 for faster inference, temperature for better quality
                result = model.transcribe(
                    audio_array,
                    fp16=True,  # Enable FP16 for CUDA
                    temperature=0.0,  # Deterministic output
                    beam_size=5,  # Balance between speed and quality
                    best_of=5,  # Multiple candidates for better accuracy
                    condition_on_previous_text=False  # Faster processing
                )
            else:
                # CPU fallback
                result = model.transcribe(audio_array, fp16=False)

            transcripts.append({
                'Participant_ID': participant_id,
                'transcript': result['text'],
                'language': result.get('language', 'unknown'),
                'confidence': result.get('avg_logprob', 0.0)
            })
            
            # Clear CUDA cache periodically to prevent memory issues
            if cuda_info['cuda_available'] and index % 10 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"WARNING: Could not process file for participant {participant_id}. Error: {e}")
            # Clear CUDA cache on error
            if cuda_info['cuda_available']:
                torch.cuda.empty_cache()

    if not transcripts:
        print("ERROR: No transcripts were generated.")
        return None

    # Create DataFrame and merge with original manifest
    transcripts_df = pd.DataFrame(transcripts)
    output_df = pd.merge(manifest_df, transcripts_df, on='Participant_ID', how='left')

    # Save results
    output_path = os.path.join(config['paths']['transcribed_data'], 'transcripts.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\nINFO: CUDA-accelerated transcription complete!")
        print(f"INFO: Output saved to {output_path}")
        print(f"INFO: Successfully transcribed {len(transcripts)} out of {len(manifest_df)} audio files.")
        
        # Print performance summary
        if cuda_info['cuda_available']:
            print(f"INFO: Processed using {cuda_info['gpu_name']} with {cuda_info['gpu_memory']:.2f} GB memory")
            
        return output_df
        
    except Exception as e:
        print(f"ERROR: Failed to save transcripts: {e}")
        return None
    finally:
        # Final CUDA cleanup
        if cuda_info['cuda_available']:
            torch.cuda.empty_cache()


def main() -> None:
    """Main function to run the CUDA-optimized data processing pipeline."""
    config_path = "config.yaml"
    
    print("="*60)
    print("DepressiVoice-LLM Data Processing (Omni Cluster Optimized)")
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
    
    # Run CUDA-accelerated transcription
    result_df = run_transcription(config, manifest_df)
    if result_df is None:
        print("ERROR: Transcription failed. Exiting.")
        return
    
    print("INFO: CUDA-optimized data processing completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()