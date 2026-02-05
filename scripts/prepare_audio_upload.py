#!/usr/bin/env python3
"""
Master Audio Upload Preparation Script
Generates SHA256 checksums and creates upload manifest
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def calculate_sha256(file_path):
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    source_dir = Path("/Users/umairtalib/Downloads/audio")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_file = f"upload_manifest_{timestamp}.json"
    
    print("\n=== MASTER AUDIO UPLOAD PREPARATION ===")
    print(f"Source directory: {source_dir}")
    
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return 1
    
    # Find all WAV files
    wav_files = sorted(source_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files\n")
    
    if len(wav_files) == 0:
        print("ERROR: No WAV files found in source directory")
        return 1
    
    # Calculate checksums
    print("Calculating SHA256 checksums...")
    files_data = []
    total_bytes = 0
    
    for wav_file in tqdm(wav_files, desc="Computing checksums"):
        size = wav_file.stat().st_size
        checksum = calculate_sha256(wav_file)
        
        files_data.append({
            "filename": wav_file.name,
            "size_bytes": size,
            "sha256_local": checksum,
            "sha256_remote": None,  # To be filled after upload
            "status": "pending"
        })
        total_bytes += size
    
    # Create manifest
    manifest = {
        "preparation_timestamp": datetime.now().isoformat(),
        "upload_timestamp": None,  # To be filled after upload
        "source": str(source_dir),
        "destination": "/work/g755389/depressivoice_master_audio",  # Will be created
        "total_files": len(files_data),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024**3), 2),
        "files": files_data,
        "success_count": 0,  # To be updated after upload
        "failure_count": 0   # To be updated after upload
    }
    
    # Save manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n=== PREPARATION COMPLETE ===")
    print(f"Total files: {len(files_data)}")
    print(f"Total size: {manifest['total_gb']} GB")
    print(f"Manifest saved to: {manifest_file}")
    print(f"\nReady for upload to cluster")
    
    return 0

if __name__ == "__main__":
    exit(main())
