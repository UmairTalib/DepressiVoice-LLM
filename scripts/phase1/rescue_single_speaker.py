#!/usr/bin/env python3
"""
Session 309 Rescue Script - Force 2 Speakers
Forces diarization to split into exactly 2 speakers, extracts both, manual review needed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

# Environment setup
if not os.environ.get('HF_TOKEN'):
    sys.exit("HF_TOKEN is not set: export your Hugging Face token first")
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model
import torch

def setup_logging(log_dir, session_id):
    """Setup logging"""
    log_file = Path(log_dir) / f"rescue_{session_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_biometric_anchor(anchor_file):
    """Load Ellie anchor"""
    with open(anchor_file, 'r') as f:
        data = json.load(f)
    return np.array(data["embedding"])

def extract_speaker_embedding(audio_file, segments, embedding_model):
    """Extract speaker embedding from segments"""
    from pyannote.audio import Inference
    
    inference = Inference(embedding_model, window="whole")
    embeddings = []
    
    for segment in segments[:10]:  # Use first 10 segments
        try:
            emb = inference.crop(audio_file, segment)
            embeddings.append(emb)
        except:
            continue
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    return None

def rescue_session_309_force_2speakers(audio_file, output_dir, anchor_file, session_id, logger):
    """
    Force diarization to detect exactly 2 speakers, then extract both for manual review
    """
    
    logger.warning("RESCUE MODE: Forcing num_speakers=2 to split single speaker")
    
    # Load models
    hf_token = os.environ.get('HF_TOKEN')
    
    logger.info("Loading Pyannote models...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    embedding_model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token=hf_token
    )
    
    # Load anchor
    logger.info(f"Loading biometric anchor from {anchor_file}")
    anchor_embedding = load_biometric_anchor(anchor_file)
    
    # FORCE 2 speakers
    logger.info(f"Running diarization with num_speakers=2 (FORCED)")
    diarization = diarization_pipeline(audio_file, num_speakers=2)
    
    speakers = list(diarization.labels())
    logger.info(f"Forced speakers: {speakers}")
    
    # Calculate similarities to anchor
    similarities = {}
    for speaker in speakers:
        segments = [seg for seg, _, lbl in diarization.itertracks(yield_label=True) if lbl == speaker]
        emb = extract_speaker_embedding(audio_file, segments, embedding_model)
        
        if emb is not None:
            # Cosine similarity
            sim = np.dot(emb.flatten(), anchor_embedding.flatten()) / (
                np.linalg.norm(emb.flatten()) * np.linalg.norm(anchor_embedding.flatten())
            )
            similarities[speaker] = float(sim)
            logger.info(f"  {speaker} similarity: {sim:.4f}")
    
    # Assign roles
    # Determine roles
    ellie_candidate = max(similarities.items(), key=lambda x: x[1])[0]
    patient_candidate = [s for s in speakers if s != ellie_candidate][0]
    
    roles = {
        ellie_candidate: 'ELLIE',
        patient_candidate: 'PATIENT'
    }
    
    logger.info(f"Assigned Roles: {roles}")
    
    # Calculate durations for metadata
    waveform, sr = sf.read(audio_file)
    audio_duration = len(waveform) / sr if len(waveform.shape) == 1 else len(waveform) / sr
    
    # Create standard metadata for roles.json
    metadata = {
        "confidence": "LOW",  # Forced split is always low confidence/manual review
        "confidence_score": 0.5,
        "margin": similarities[ellie_candidate] - similarities[patient_candidate],
        "duration_ratio": 1.0,  # Placeholder
        "ellie_duration": 0.0,   # Placeholder
        "patient_duration": 0.0  # Placeholder
    }
    
    # Save standard outputs (RTTM and roles.json)
    save_diarization_results(diarization, roles, similarities, metadata, output_dir, session_id)
    
    # Load audio for extraction
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Create output directory (already created in save_diarization_results but safe to keep)
    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract standard patient_audio.wav (auto-selected based on roles)
    patient_segments = []
    for seg, _, lbl in diarization.itertracks(yield_label=True):
        if lbl == patient_candidate:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            if start_sample < len(waveform) and end_sample <= len(waveform):
                patient_segments.append(waveform[start_sample:end_sample])
    
    if patient_segments:
        patient_audio = np.concatenate(patient_segments)
        output_file = session_dir / "patient_audio.wav"
        sf.write(output_file, patient_audio, sr)
        logger.info(f"Saved standard patient audio: {output_file}")
    
    # Also save separate candidate files for manual verification if needed
    for speaker in speakers:
        segments = []
        for seg, _, lbl in diarization.itertracks(yield_label=True):
            if lbl == speaker:
                start_sample = int(seg.start * sr)
                end_sample = int(seg.end * sr)
                if start_sample < len(waveform) and end_sample <= len(waveform):
                    segments.append(waveform[start_sample:end_sample])
        
        if segments:
            speaker_audio = np.concatenate(segments)
            role_suffix = "ELLIE" if speaker == ellie_candidate else "PATIENT"
            output_file = session_dir / f"{speaker}_{role_suffix}.wav"
            sf.write(output_file, speaker_audio, sr)
            logger.info(f"Saved candidate audio: {output_file}")

    # Create quality flag for forced split
    quality_flag = {
        "session_id": session_id,
        "flag_type": "FORCED_2SPEAKER_SPLIT",
        "confidence": "MANUAL_REVIEW",
        "reason": "Only 1 speaker detected originally - forced split into 2",
        "review_recommended": True
    }
    
    with open(session_dir / "quality_flag.json", 'w') as f:
        json.dump(quality_flag, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"RESCUE COMPLETE for Session {session_id}")
    logger.info(f"Generated:")
    logger.info(f"  - {session_id}.rttm")
    logger.info(f"  - roles.json")
    logger.info(f"  - patient_audio.wav")
    logger.info("=" * 60)

def save_diarization_results(diarization, roles, similarities, metadata, output_dir, session_id):
    """Save diarization results to RTTM and roles JSON ensuring compatibility"""
    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save RTTM
    rttm_file = session_dir / f"{session_id}.rttm"
    with open(rttm_file, 'w') as f:
        for segment, _, label in diarization.itertracks(yield_label=True):
            f.write(f"SPEAKER {session_id} 1 {segment.start:.3f} {segment.duration:.3f} <NA> <NA> {label} <NA> <NA>\n")
    
    # Save roles.json
    roles_file = session_dir / "roles.json"
    roles_data = {
        "roles": roles,
        "similarities": similarities,
        "separation_margin": metadata.get("margin", 0.0),
        "confidence": "LOW",
        "note": "Generated via rescue_single_speaker.py (forced split)",
        "durations": {
            "ellie": metadata.get("ellie_duration", 0.0),
            "patient": metadata.get("patient_duration", 0.0)
        }
    }
    
    with open(roles_file, 'w') as f:
        json.dump(roles_data, f, indent=2)
    
    return str(rttm_file), str(roles_file)

def main():
    parser = argparse.ArgumentParser(description="Rescue session 309 with forced 2-speaker split")
    parser.add_argument("--session_id", default="309", help="Session ID")
    parser.add_argument("--audio_file", 
                       default="/home/g755389/depressivoice_master_audio/309_AUDIO.wav",
                       help="Master audio file")
    parser.add_argument("--output_dir",
                       default="/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs",
                       help="Output directory")
    parser.add_argument("--anchor_file",
                       default="/home/g755389/DepressiVoice-LLM/biometric_anchor.pt",
                       help="Ellie biometric anchor file")
    parser.add_argument("--log_dir",
                       default="/home/g755389/DepressiVoice-LLM/logs",
                       help="Log directory")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.log_dir, args.session_id)
    
    print(f"\n{'='*60}")
    print(f"SESSION {args.session_id} RESCUE - FORCE 2 SPEAKERS")
    print(f"{'='*60}\n")
    
    logger.info(f"Starting rescue for session {args.session_id}")
    
    # Run rescue
    rescue_session_309_force_2speakers(
        args.audio_file,
        args.output_dir,
        args.anchor_file,
        args.session_id,
        logger
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
