#!/usr/bin/env python3
"""
Phase 1 Diarization Pipeline - Complete Rerun
Performs speaker diarization, role assignment, and audio extraction
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Annotation, Segment
import logging

def setup_logging(log_dir, session_id):
    """Setup logging for this session"""
    log_file = Path(log_dir) / f"session_{session_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_biometric_anchor(anchor_path):
    """Load the pre-computed Ellie biometric anchor"""
    if not Path(anchor_path).exists():
        raise FileNotFoundError(f"Anchor not found: {anchor_path}")
    
    with open(anchor_path, 'r') as f:
        anchor_data = json.load(f)
    
    return np.array(anchor_data['embedding'])

def extract_speaker_embedding(audio_file, segments, embedding_model):
    """Extract speaker embedding from audio segments"""
    if not segments:
        return None
    
    # Load audio
    waveform, sr = sf.read(audio_file)
    
    # Handle stereo (convert to mono)
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Concatenate all segments for this speaker
    speaker_audio = []
    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        if start_sample < len(waveform) and end_sample <= len(waveform):
            speaker_audio.append(waveform[start_sample:end_sample])
    
    if not speaker_audio:
        return None
    
    # Concatenate and convert to tensor
    speaker_audio = np.concatenate(speaker_audio)
    waveform_tensor = torch.from_numpy(speaker_audio).float()
    
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
    
    # Get embedding
    with torch.no_grad():
        embedding = embedding_model(waveform_tensor)
    
    if hasattr(embedding, 'detach'):
        return embedding.detach().cpu().numpy()
    return embedding

def assign_roles(diarization, audio_file, embedding_model, anchor_embedding, logger):
    """Assign ELLIE/PATIENT roles using biometric anchor"""
    
    # Extract embeddings for each speaker
    speaker_embeddings = {}
    similarities = {}
    
    for label in diarization.labels():
        segments = [seg for seg, _, lbl in diarization.itertracks(yield_label=True) if lbl == label]
        embedding = extract_speaker_embedding(audio_file, segments, embedding_model)
        
        if embedding is not None:
            speaker_embeddings[label] = embedding
            # Cosine similarity with anchor
            similarity = np.dot(embedding.flatten(), anchor_embedding.flatten()) / (
                np.linalg.norm(embedding.flatten()) * np.linalg.norm(anchor_embedding.flatten())
            )
            similarities[label] = float(similarity)
    
    if not similarities:
        logger.error("No speaker embeddings could be extracted")
        return None, None, None
    
    # Assign roles: highest similarity = ELLIE
    ellie_label = max(similarities.items(), key=lambda x: x[1])[0]
    
    # Calculate total speaking duration for ALL speakers (needed for logging)
    all_durations = {}
    for label in similarities.keys():
        total_duration = sum([
            seg.duration 
            for seg, _, lbl in diarization.itertracks(yield_label=True) 
            if lbl == label
        ])
        all_durations[label] = total_duration
    
    # Patient is the OTHER speaker who speaks most (by duration)
    other_speakers = {k: v for k, v in similarities.items() if k != ellie_label}
    
    if not other_speakers:
        logger.error("Only one speaker found - cannot assign patient role")
        return None, None, None
    
    # Get durations for non-Ellie speakers only
    other_durations = {k: all_durations[k] for k in other_speakers.keys()}
    
    # Patient is the speaker with longest duration (excluding Ellie)
    patient_label = max(other_durations.items(), key=lambda x: x[1])[0]
    
    roles = {
        ellie_label: 'ELLIE',
        patient_label: 'PATIENT'
    }
    
    # Calculate separation margin (similarity difference)
    margin = similarities[ellie_label] - similarities[patient_label]
    
    # Confidence scoring based on margin
    if margin >= 0.2:
        confidence = "HIGH"
        confidence_score = 1.0
    elif margin >= 0.1:
        confidence = "MEDIUM"
        confidence_score = 0.7
    elif margin >= 0.05:
        confidence = "LOW"
        confidence_score = 0.5
        logger.warning(f"Low separation margin ({margin:.4f}) - likely same-gender speakers")
    else:
        confidence = "VERY_LOW"
        confidence_score = 0.3
        logger.warning(f"Very low separation margin ({margin:.4f}) - MANUAL REVIEW RECOMMENDED")
    
    # Duration ratio for additional confidence
    duration_ratio = all_durations[patient_label] / all_durations[ellie_label] if all_durations[ellie_label] > 0 else 0
    
    logger.info(f"Role Assignment:")
    logger.info(f"  ELLIE: {ellie_label} (similarity: {similarities[ellie_label]:.4f}, duration: {all_durations[ellie_label]:.2f}s)")
    logger.info(f"  PATIENT: {patient_label} (similarity: {similarities[patient_label]:.4f}, duration: {all_durations[patient_label]:.2f}s)")
    logger.info(f"  Separation Margin: {margin:.4f}")
    logger.info(f"  Confidence: {confidence} (score: {confidence_score:.2f})")
    logger.info(f"  Duration Ratio (Patient/Ellie): {duration_ratio:.2f}")
    
    # Create metadata dict
    metadata = {
        "confidence": confidence,
        "confidence_score": confidence_score,
        "margin": margin,
        "duration_ratio": duration_ratio,
        "ellie_duration": all_durations[ellie_label],
        "patient_duration": all_durations[patient_label]
    }
    
    return roles, similarities, metadata

def extract_patient_audio(audio_file, diarization, roles, output_dir, session_id, logger):
    """Extract patient audio segments"""
    
    # Find patient label
    patient_label = next((lbl for lbl, role in roles.items() if role == 'PATIENT'), None)
    
    if not patient_label:
        logger.error("No PATIENT role assigned")
        return None
    
    # Load audio
    waveform, sr = sf.read(audio_file)
    
    # Handle stereo
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Extract patient segments
    patient_segments = []
    for seg, _, lbl in diarization.itertracks(yield_label=True):
        if lbl == patient_label:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            if start_sample < len(waveform) and end_sample <= len(waveform):
                patient_segments.append(waveform[start_sample:end_sample])
    
    if not patient_segments:
        logger.error("No patient audio segments found")
        return None
    
    # Concatenate patient audio
    patient_audio = np.concatenate(patient_segments)
    
    # Save patient audio
    output_file = Path(output_dir) / f"session_{session_id}" / "patient_audio.wav"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, patient_audio, sr)
    
    logger.info(f"Patient audio saved: {output_file} ({len(patient_audio)/sr:.2f} seconds)")
    
    return str(output_file)

def save_diarization_results(diarization, roles, similarities, metadata, output_dir, session_id):
    """Save diarization results to RTTM and roles JSON with confidence scoring"""
    
    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save RTTM
    rttm_file = session_dir / f"{session_id}.rttm"
    with open(rttm_file, 'w') as f:
        for segment, _, label in diarization.itertracks(yield_label=True):
            # RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <label> <NA> <NA>
            f.write(f"SPEAKER {session_id} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {label} <NA> <NA>\n")
    
    # Save roles with metadata
    roles_file = session_dir / "roles.json"
    roles_data = {
        "roles": roles,
        "similarities": similarities,
        "separation_margin": metadata["margin"],
        "confidence": metadata["confidence"],
        "confidence_score": metadata["confidence_score"],
        "duration_ratio": metadata["duration_ratio"],
        "durations": {
            "ellie": metadata["ellie_duration"],
            "patient": metadata["patient_duration"]
        }
    }
    
    with open(roles_file, 'w') as f:
        json.dump(roles_data, f, indent=2)
    
    # Save quality flag if low confidence
    if metadata["confidence"] in ["LOW", "VERY_LOW"]:
        quality_flag_file = session_dir / "quality_flag.json"
        flag_data = {
            "session_id": session_id,
            "flag_type": "LOW_SEPARATION_MARGIN",
            "confidence": metadata["confidence"],
            "margin": metadata["margin"],
            "reason": "Low biometric separation - likely same-gender speakers (female-female)",
            "review_recommended": metadata["confidence"] == "VERY_LOW",
            "duration_ratio": metadata["duration_ratio"]
        }
        
        with open(quality_flag_file, 'w') as f:
            json.dump(flag_data, f, indent=2)
    
    return str(rttm_file), str(roles_file)

def run_diarization(args):
    """Main diarization function"""
    
    logger = setup_logging(args.log_dir, args.session_id)
    logger.info(f"Starting diarization for session {args.session_id}")
    
    # Check audio file exists
    if not Path(args.audio_file).exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        return 1
    
    # Load HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        return 1
    
    # Load models
    logger.info("Loading Pyannote models...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    embedding_model = PretrainedSpeakerEmbedding(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token=hf_token
    )
    
    # Load biometric anchor
    anchor_path = Path(args.output_dir).parent / "ellie_anchor.json"
    logger.info(f"Loading biometric anchor from {anchor_path}")
    
    try:
        anchor_embedding = load_biometric_anchor(anchor_path)
    except FileNotFoundError:
        logger.warning("Anchor not found - will be created by anchor generation script first")
        return 2  # Special exit code indicating anchor needed
    
    # Run diarization
    logger.info(f"Running diarization on {args.audio_file}")
    diarization = diarization_pipeline(args.audio_file)
    
    logger.info(f"Found {len(diarization.labels())} speakers")
    
    # Assign roles
    roles, similarities, metadata = assign_roles(
        diarization, args.audio_file, embedding_model, anchor_embedding, logger
    )
    
    if roles is None:
        logger.error("Role assignment failed")
        return 1
    
    # Save diarization results
    rttm_file, roles_file = save_diarization_results(
        diarization, roles, similarities, metadata, args.output_dir, args.session_id
    )
    
    logger.info(f"Saved RTTM: {rttm_file}")
    logger.info(f"Saved roles: {roles_file}")
    
    # Extract patient audio
    patient_audio_file = extract_patient_audio(
        args.audio_file, diarization, roles, args.output_dir, args.session_id, logger
    )
    
    if patient_audio_file:
        logger.info(f"Saved patient audio: {patient_audio_file}")
    
    logger.info(f"Session {args.session_id} completed successfully")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Diarization Pipeline")
    parser.add_argument("--session_id", required=True, help="Session ID")
    parser.add_argument("--audio_file", required=True, help="Path to master audio WAV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--log_dir", required=True, help="Log directory")
    
    args = parser.parse_args()
    
    exit_code = run_diarization(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
