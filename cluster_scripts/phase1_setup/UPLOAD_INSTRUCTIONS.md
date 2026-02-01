# Upload Instructions for Omni HPC Cluster

## Your Cluster Credentials

**Username**: `g755389`  
**Hostname**: `omni.zimt.uni-siegen.de`  
**SSH Command**: `ssh g755389@omni.zimt.uni-siegen.de`

## Option 1: Using rsync (Recommended)

```bash
# From your local machine, in the repository root:
cd /Users/umairtalib/DepressiVoice-LLM

# Upload orchestration scripts to cluster
rsync -avz --progress \
  cluster_scripts/phase1_setup/ \
  g755389@omni.zimt.uni-siegen.de:~/phase1_orchestration/
```

## Option 2: Using scp

```bash
# From your local machine, in the repository root:
cd /Users/umairtalib/DepressiVoice-LLM

# Upload entire directory
scp -r cluster_scripts/phase1_setup \
  g755389@omni.zimt.uni-siegen.de:~/phase1_orchestration/
```

## Verify Upload

```bash
# SSH into the cluster
ssh g755389@omni.zimt.uni-siegen.de

# Check files uploaded
ls -lh ~/phase1_orchestration/

# Expected files:
# - 00_collect_info.sh
# - 01_precheck.sh
# - 02_create_worktree.sh
# - 03_env_check.sbatch.template
# - 04_monitor_job.sh
# - 05_finalize.sh
# - generate_batch_script.sh
# - requirements_phase1.txt
# - validate_imports.py
# - README.md
# - UPLOAD_INSTRUCTIONS.md (this file)
```

## Make Scripts Executable

```bash
# On the cluster:
cd ~/phase1_orchestration
chmod +x *.sh
chmod +x validate_imports.py

# Verify
ls -lh
```

## Next Steps

After uploading, follow the Quick Start guide in README.md:

```bash
# On the cluster:
cd ~/phase1_orchestration
./00_collect_info.sh
```

## Alternative: Direct Clone on Cluster

If your cluster has internet access:

```bash
# On the cluster (after: ssh g755389@omni.zimt.uni-siegen.de):
cd ~
git clone https://github.com/UmairTalib/DepressiVoice-LLM.git temp_clone
cp -r temp_clone/cluster_scripts/phase1_setup ~/phase1_orchestration
rm -rf temp_clone
cd ~/phase1_orchestration
chmod +x *.sh
./00_collect_info.sh
```

## Troubleshooting Upload

**Permission denied / SSH key issues:**
```bash
# Generate SSH key if needed (on your local machine)
ssh-keygen -t ed25519 -C "umair@depressivoice"

# Add to cluster
ssh-copy-id g755389@omni.zimt.uni-siegen.de
```

**Connection timeout:**
- Verify VPN connection if cluster requires it
- Hostname is: `omni.zimt.uni-siegen.de`
- Contact cluster support if connection still fails

**Files not appearing:**
- Check target directory: `ls -la ~/phase1_orchestration/`
- Verify no typos in username/hostname
- Try absolute path: `~/phase1_orchestration` vs `phase1_orchestration`
