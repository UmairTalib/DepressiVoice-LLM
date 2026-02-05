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
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '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')
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
    ellie_candidate = max(similarities.items(), key=lambda x: x[1])[0]
    patient_candidate = [s for s in speakers if s != ellie_candidate][0]
    
    logger.info(f"Tentative assignment: ELLIE={ellie_candidate}, PATIENT={patient_candidate}")
    
    # Load audio
    waveform, sr = sf.read(audio_file)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Create output directory
    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract BOTH speakers for manual review
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
            duration = len(speaker_audio) / sr
            
            # Determine if this is the patient candidate
            if speaker == patient_candidate:
                output_file = session_dir / f"patient_audio_CANDIDATE.wav"
                role = "PATIENT_CANDIDATE"
            else:
                output_file = session_dir / f"ellie_audio_CANDIDATE.wav"
                role = "ELLIE_CANDIDATE"
            
            sf.write(output_file, speaker_audio, sr)
            logger.info(f"Extracted {speaker} as {role}: {output_file.name} ({duration:.2f}s)")
    
    # Save rescue metadata
    rescue_metadata = {
        "session_id": session_id,
        "rescue_mode": "FORCE_2_SPEAKERS",
        "original_speakers_detected": 1,
        "forced_speakers": 2,
        "tentative_roles": {
            ellie_candidate: "ELLIE_CANDIDATE",
            patient_candidate: "PATIENT_CANDIDATE"
        },
        "similarities": similarities,
        "manual_review_required": True,
        "instruction": "Listen to BOTH audio files and manually select the correct patient audio"
    }
    
    rescue_file = session_dir / "rescue_metadata.json"
    with open(rescue_file, 'w') as f:
        json.dump(rescue_metadata, f, indent=2)
    
    logger.info(f"Rescue metadata saved: {rescue_file}")
    
    # Create quality flag
    quality_flag = {
        "session_id": session_id,
        "flag_type": "FORCED_2SPEAKER_SPLIT",
        "confidence": "MANUAL_REVIEW",
        "reason": "Only 1 speaker detected originally - forced split into 2",
        "review_recommended": True,
        "action_required": "Listen to both candidates and pick correct patient audio"
    }
    
    quality_flag_file = session_dir / "quality_flag.json"
    with open(quality_flag_file, 'w') as f:
        json.dump(quality_flag, f, indent=2)
    
    logger.warning("=" * 60)
    logger.warning("MANUAL REVIEW REQUIRED")
    logger.warning(f"Files created:")
    logger.warning(f"  - patient_audio_CANDIDATE.wav (tentative patient)")
    logger.warning(f"  - ellie_audio_CANDIDATE.wav (tentative interviewer)")
    logger.warning("")
    logger.warning("ACTION: Listen to BOTH files and:")
    logger.warning("  1. Rename the correct one to 'patient_audio.wav'")
    logger.warning("  2. Delete the other file")
    logger.warning("=" * 60)

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
                       default="/home/g755389/DepressiVoice-LLM/ellie_anchor.json",
                       help="Ellie biometric anchor file")
    parser.add_argument("--log_dir",
                       default="/home/g755389/DepressiVoice-LLM/logs",
                       help="Log directory")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_dir, args.session_id)
    
    print(f"\n{'='*60}")
    print(f"SESSION 309 RESCUE - FORCE 2 SPEAKERS")
    print(f"{'='*60}\n")
    
    logger.info(f"Starting rescue for session {args.session_id}")
    logger.info(f"Input: {args.audio_file}")
    
    # Run rescue
    rescue_session_309_force_2speakers(
        args.audio_file,
        args.output_dir,
        args.anchor_file,
        args.session_id,
        logger
    )
    
    print(f"\n{'='*60}")
    print(f"RESCUE COMPLETE - MANUAL REVIEW REQUIRED")
    print(f"Check: {args.output_dir}/session_{args.session_id}/")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
