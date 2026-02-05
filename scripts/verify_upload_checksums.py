#!/usr/bin/env python3
"""
Remote Checksum Verification Script
Computes SHA256 checksums on cluster and compares with local manifest
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

def main():
    # Load local manifest
    manifest_files = sorted(Path(".").glob("upload_manifest_*.json"))
    if not manifest_files:
        print("ERROR: No upload manifest found")
        return 1
    
    manifest_file = manifest_files[-1]  # Use latest
    print(f"Loading manifest: {manifest_file}")
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    cluster_dest = "/home/g755389/depressivoice_master_audio"
    
    print(f"\n=== REMOTE CHECKSUM VERIFICATION ===")
    print(f"Verifying {len(manifest['files'])} files on cluster...")
    
    # Generate remote checksums
    print("\nComputing SHA256 checksums on cluster (this may take 5-10 minutes)...")
    
    cmd = f"ssh g755389@omni.zimt.uni-siegen.de 'cd {cluster_dest} && shasum -a 256 *.wav'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR running remote checksum: {result.stderr}")
        return 1
    
    # Parse remote checksums
    remote_checksums = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            checksum, filename = parts[0], parts[1]
            remote_checksums[filename] = checksum
    
    print(f"Computed {len(remote_checksums)} remote checksums")
    
    # Compare and update manifest
    success_count = 0
    failure_count = 0
    
    for file_info in manifest['files']:
        filename = file_info['filename']
        local_checksum = file_info['sha256_local']
        
        if filename in remote_checksums:
            remote_checksum = remote_checksums[filename]
            file_info['sha256_remote'] = remote_checksum
            
            if local_checksum == remote_checksum:
                file_info['status'] = 'verified'
                success_count += 1
            else:
                file_info['status'] = 'checksum_mismatch'
                failure_count += 1
                print(f"  ✗ MISMATCH: {filename}")
        else:
            file_info['status'] = 'missing'
            failure_count += 1
            print(f"  ✗ MISSING: {filename}")
    
    # Update manifest
    manifest['upload_timestamp'] = datetime.now().isoformat()
    manifest['success_count'] = success_count
    manifest['failure_count'] = failure_count
    manifest['destination'] = cluster_dest  # Update actual destination
    
    # Save updated manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    print(f"Successfully verified: {success_count}")
    print(f"Failed/Missing: {failure_count}")
    print(f"Updated manifest: {manifest_file}")
    
    if failure_count > 0:
        print("\n⚠ WARNING: Some files failed verification!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
