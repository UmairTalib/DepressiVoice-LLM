#!/usr/bin/env python3
"""
Phase 1 Transcription Pipeline - Faster-Whisper
Transcribes patient audio segments with precise timestamps
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from faster_whisper import WhisperModel
import soundfile as sf

def setup_logging(log_dir, session_id):
    """Setup logging for this session"""
    log_file = Path(log_dir) / f"transcription_{session_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def transcribe_patient_audio(audio_file, model, logger):
    """Transcribe patient audio using faster-whisper"""
    
    logger.info(f"Transcribing {audio_file}")
    
    # Load audio to check duration
    waveform, sr = sf.read(audio_file)
    duration = len(waveform) / sr
    logger.info(f"Audio duration: {duration:.2f} seconds")
    
    # Transcribe with faster-whisper
    segments, info = model.transcribe(
        str(audio_file),
        language="en",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )
    
    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Collect segments
    transcript_segments = []
    for segment in segments:
        segment_data = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "words": []
        }
        
        # Add word-level timestamps if available
        if segment.words:
            for word in segment.words:
                segment_data["words"].append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                })
        
        transcript_segments.append(segment_data)
        logger.debug(f"Segment {segment.id}: [{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}")
    
    logger.info(f"Transcription complete: {len(transcript_segments)} segments")
    
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": duration,
        "segments": transcript_segments
    }

def save_transcript(transcript_data, output_dir, session_id, logger):
    """Save transcript to JSON file"""
    
    session_dir = Path(output_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_file = session_dir / "transcript.json"
    
    with open(transcript_file, 'w') as f:
        json.dump(transcript_data, f, indent=2)
    
    logger.info(f"Transcript saved: {transcript_file}")
    
    # Also save plain text version
    text_file = session_dir / "patient_text.txt"
    full_text = " ".join([seg["text"] for seg in transcript_data["segments"]])
    
    with open(text_file, 'w') as f:
        f.write(full_text)
    
    logger.info(f"Plain text saved: {text_file}")
    
    return str(transcript_file)

def run_transcription(args):
    """Main transcription function"""
    
    logger = setup_logging(args.log_dir, args.session_id)
    logger.info(f"Starting transcription for session {args.session_id}")
    
    # Check patient audio exists
    patient_audio = Path(args.output_dir) / f"session_{args.session_id}" / "patient_audio.wav"
    
    if not patient_audio.exists():
        logger.error(f"Patient audio not found: {patient_audio}")
        logger.error("Make sure diarization has completed for this session")
        return 1
    
    # Load faster-whisper model
    logger.info(f"Loading faster-whisper model: {args.model}")
    model = WhisperModel(
        args.model,
        device="cuda",
        compute_type="float16",
        download_root=args.cache_dir
    )
    
    # Transcribe
    transcript_data = transcribe_patient_audio(patient_audio, model, logger)
    
    # Save results
    transcript_file = save_transcript(
        transcript_data, args.output_dir, args.session_id, logger
    )
    
    logger.info(f"Session {args.session_id} transcription completed successfully")
    logger.info(f"Total segments: {len(transcript_data['segments'])}")
    logger.info(f"Duration: {transcript_data['duration']:.2f} seconds")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Transcription Pipeline")
    parser.add_argument("--session_id", required=True, help="Session ID")
    parser.add_argument("--output_dir", required=True, help="Output directory (same as diarization)")
    parser.add_argument("--log_dir", required=True, help="Log directory")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--cache_dir", default="/home/g755389/.cache/whisper", help="Model cache directory")
    
    args = parser.parse_args()
    
    exit_code = run_transcription(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
