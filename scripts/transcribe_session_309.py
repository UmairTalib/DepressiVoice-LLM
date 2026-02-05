#!/usr/bin/env python3
"""
Transcribe Session 309 - Simple wrapper
"""

import sys
import os

# Add src to path
sys.path.insert(0, '/home/g755389/DepressiVoice-LLM/src')

# Set environment
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

from phase1_transcription_rerun import run_transcription

def main():
    # Create args namespace
    class Args:
        pass
    
    args = Args()
    args.session_id = "309"
    args.output_dir = "/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs"
    args.log_dir = "/home/g755389/DepressiVoice-LLM/logs"
    args.model = "large-v3"
    args.cache_dir = "/home/g755389/.cache/whisper"
    
    print(f"\n{'='*60}")
    print(f"Transcribing Session 309")
    print(f"{'='*60}\n")
    
    exit_code = run_transcription(args)
    
    if exit_code == 0:
        print(f"\n{'='*60}")
        print(f"SUCCESS - Transcription Complete")
        print(f"Output: {args.output_dir}/session_309/transcript.json")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"FAILED with exit code {exit_code}")
        print(f"{'='*60}\n")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
