# Phase-1 HPC Environment Setup - Orchestration

Complete orchestration system for setting up a safe Phase-1 compute environment on Omni HPC cluster.

## Quick Start (On Cluster)

```bash
# 1. Upload this directory to your cluster
# 2. Run configuration collector
./00_collect_info.sh

# 3. Review and edit cluster_config.env if needed
nano cluster_config.env

# 4. Source configuration
source cluster_config.env

# 5. Run precheck
./01_precheck.sh

# 6. Create safe worktree
./02_create_worktree.sh

# 7. Generate and submit SLURM job
./generate_batch_script.sh
sbatch 03_env_check.sbatch

# 8. Monitor progress
./04_monitor_job.sh

# 9. Finalize and review
./05_finalize.sh

# 10. Approve for Phase-1 (after reviewing reports)
echo '{"approved": true}' > $OUT_ROOT/env_install_logs/human_ok.json
```

## Scripts Overview

| Script | Purpose | Runs On |
|--------|---------|---------|
| `00_collect_info.sh` | Gathers cluster configuration interactively | Login node |
| `01_precheck.sh` | Validates environment, creates log structure | Login node |
| `02_create_worktree.sh` | Creates safe git worktree/copy without data | Login node |
| `generate_batch_script.sh` | Generates SLURM script from template | Login node |
| `03_env_check.sbatch` | Installs packages, runs validation | **Compute node** |
| `validate_imports.py` | Checks all package imports and CUDA | **Compute node** |
| `04_monitor_job.sh` | Monitors SLURM job progress | Login node |
| `05_finalize.sh` | Creates env_ready.json or failure report | Login node |

## Safety Guarantees

✅ **Original repository (REPO_ROOT) remains read-only** - no modifications  
✅ **All work happens in separate worktree** under OUT_ROOT  
✅ **No patient data copied** - audio/CSV files excluded  
✅ **No GPU operations on login node** - all heavy work via SLURM  
✅ **Human approval required** before Phase-1 execution  
✅ **Complete audit trail** in actions.log  

## Output Structure

After successful completion:

```
$OUT_ROOT/
├── phase1_worktree/              # Safe copy of code (no data)
└── env_install_logs/
    ├── actions.log               # Complete command log
    ├── cluster_config.env        # Your configuration
    ├── precheck_result.json      # Environment validation
    ├── worktree_manifest.json    # File hashes
    ├── compute_imports.json      # Import check results
    ├── torch_cuda.json           # PyTorch/CUDA details
    ├── pip_list_compute.json     # All installed packages
    ├── ffmpeg_check.json         # FFmpeg availability
    ├── compute_result.json       # Overall status (PASS/WARN/FAIL)
    ├── env_ready.json            # Ready marker (on success)
    ├── env_install_summary.txt   # Human-readable summary
    └── human_ok.json             # Approval marker (you create this)
```

## Validation Status

- **PASS**: All critical packages working, ready for Phase-1
- **WARN**: Core packages OK, some optional packages missing (review warnings)
- **FAIL**: Critical packages failed, cannot proceed (see failure_report.zip)

## Critical Packages Checked

**Required for PASS:**
- pandas, numpy (data processing)
- torch (deep learning)
- ffmpeg OR torchaudio (audio processing)

**Optional (WARN if missing):**
- librosa (audio features)
- pyannote.audio (speaker diarization)
- whisperx (transcription)

**Not critical for Phase-1:**
- CUDA (nice to have, not required for preprocessing)

## Troubleshooting

**If validation fails:**
1. Check `compute_result.json` for specific errors
2. Review SLURM output: `cat phase1_env_check_*.out`
3. Check failure report: `unzip failure_report_*.zip`
4. Common fixes:
   - Network timeout: Re-run `sbatch 03_env_check.sbatch`
   - Missing module: Edit `cluster_config.env`, reload conda module
   - Package conflict: Edit `requirements_phase1.txt`, remove conflicting versions

**If worktree creation fails:**
1. Check git version: `git --version` (need 2.5+)
2. Use fallback: Automatic selective copy will trigger
3. Verify exclusions: `find $OUT_ROOT/phase1_worktree -name "*.wav"` (should be empty)

## Next Steps After Approval

Once `human_ok.json` exists, you can proceed with Phase-1 workflows:
- Data preprocessing
- Speaker diarization
- Transcription
- Quality validation

All work should happen in `$OUT_ROOT/phase1_worktree`, never in REPO_ROOT.

## Support

For issues or questions:
1. Check `env_install_logs/actions.log` for detailed command history
2. Review JSON reports for specific error messages
3. Verify REPO_ROOT unchanged: `cd $REPO_ROOT && git status` (should be clean)
