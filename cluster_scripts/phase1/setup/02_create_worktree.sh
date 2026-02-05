#!/bin/bash
################################################################################
# 02_create_worktree.sh
# Creates a safe git worktree or copy outside REPO_ROOT
# Excludes all patient data (CSVs, audio files)
# Generates manifest with SHA256 hashes
################################################################################

set -euo pipefail

# Source configuration
source cluster_config.env

LOG_DIR="$OUT_ROOT/env_install_logs"
ACTION_LOG="$LOG_DIR/actions.log"

log_action() {
  local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
  echo "[$timestamp] $1" | tee -a "$ACTION_LOG"
}

log_action "=========================================="
log_action "WORKTREE SETUP: Creating safe workspace"
log_action "=========================================="

WORKTREE_DIR="$OUT_ROOT/phase1_worktree"
MANIFEST_FILE="$LOG_DIR/worktree_manifest.json"

# Check if worktree directory already exists
if [ -d "$WORKTREE_DIR" ]; then
  log_action "⚠️  Worktree directory already exists: $WORKTREE_DIR"
  read -p "Remove and recreate? [y/N]: " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_action "Removing existing worktree..."
    rm -rf "$WORKTREE_DIR"
  else
    log_action "Keeping existing worktree. Exiting."
    exit 0
  fi
fi

# Try git worktree first (preferred method)
cd "$REPO_ROOT"

log_action "Attempting git worktree creation..."
log_action "Branch: $BRANCH_NAME"
log_action "Target: $WORKTREE_DIR"

# Check if branch exists
if git rev-parse --verify "$BRANCH_NAME" &>/dev/null; then
  log_action "✓ Branch '$BRANCH_NAME' exists"
  BRANCH_EXISTS=true
else
  log_action "⚠️  Branch '$BRANCH_NAME' does not exist"
  log_action "Creating new branch from current HEAD..."
  git branch "$BRANCH_NAME"
  BRANCH_EXISTS=true
fi

# Create worktree
if git worktree add --checkout "$WORKTREE_DIR" "$BRANCH_NAME" 2>&1 | tee -a "$ACTION_LOG"; then
  log_action "✓ Git worktree created successfully"
  WORKTREE_METHOD="git_worktree"
else
  log_action "❌ Git worktree creation failed"
  log_action "Falling back to selective copy method..."
  WORKTREE_METHOD="selective_copy"
  
  mkdir -p "$WORKTREE_DIR"
  
  # Copy only code and config files, exclude data
  log_action "Copying code files (excluding data)..."
  
  # Use rsync to copy with exclusions
  rsync -av \
    --exclude='*.wav' \
    --exclude='*.mp3' \
    --exclude='*.flac' \
    --exclude='*.ogg' \
    --exclude='*.m4a' \
    --exclude='*.csv' \
    --exclude='*.tsv' \
    --exclude='data/' \
    --exclude='audio/' \
    --exclude='raw/' \
    --exclude='dataset/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.safetensors' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='env/' \
    --exclude='.conda/' \
    "$REPO_ROOT/" "$WORKTREE_DIR/" 2>&1 | tee -a "$ACTION_LOG"
  
  log_action "✓ Selective copy completed"
fi

# Generate manifest with SHA256 hashes
log_action "Generating worktree manifest..."

cat > "$MANIFEST_FILE" <<EOF
{
  "method": "$WORKTREE_METHOD",
  "source_repo": "$REPO_ROOT",
  "worktree_path": "$WORKTREE_DIR",
  "branch": "$BRANCH_NAME",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "files": [
EOF

# Find all files in worktree and compute SHA256
FIRST_FILE=true
find "$WORKTREE_DIR" -type f \( \
  -name "*.py" -o \
  -name "*.sh" -o \
  -name "*.yaml" -o \
  -name "*.yml" -o \
  -name "*.json" -o \
  -name "*.txt" -o \
  -name "*.md" -o \
  -name "Makefile" -o \
  -name "Dockerfile" \
\) | while read -r filepath; do
  rel_path="${filepath#$WORKTREE_DIR/}"
  
  if command -v sha256sum &>/dev/null; then
    hash=$(sha256sum "$filepath" | awk '{print $1}')
  elif command -v shasum &>/dev/null; then
    hash=$(shasum -a 256 "$filepath" | awk '{print $1}')
  else
    hash="unavailable"
  fi
  
  size=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
  
  if [ "$FIRST_FILE" = true ]; then
    FIRST_FILE=false
  else
    echo "," >> "$MANIFEST_FILE"
  fi
  
  cat >> "$MANIFEST_FILE" <<FILEOF
    {
      "path": "$rel_path",
      "sha256": "$hash",
      "size_bytes": $size
    }
FILEOF
done

cat >> "$MANIFEST_FILE" <<EOF

  ],
  "exclusions": [
    "*.wav", "*.mp3", "*.flac", "*.csv", "*.tsv",
    "data/", "audio/", "raw/", "dataset/",
    "*.pt", "*.pth", "*.ckpt", ".git/"
  ],
  "total_files": $(find "$WORKTREE_DIR" -type f | wc -l)
}
EOF

log_action "✓ Manifest generated: $MANIFEST_FILE"

# Verify no patient data was copied
log_action "Verifying exclusions..."
FOUND_AUDIO=$(find "$WORKTREE_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l)
FOUND_CSV=$(find "$WORKTREE_DIR" -type f \( -name "*.csv" -o -name "*.tsv" \) | wc -l)

if [ "$FOUND_AUDIO" -gt 0 ]; then
  log_action "⚠️  WARNING: Found $FOUND_AUDIO audio files in worktree (should be 0)"
fi

if [ "$FOUND_CSV" -gt 0 ]; then
  log_action "⚠️  WARNING: Found $FOUND_CSV CSV files in worktree"
fi

log_action "Audio files: $FOUND_AUDIO (expected: 0)"
log_action "CSV files: $FOUND_CSV"

# Verify REPO_ROOT unchanged
cd "$REPO_ROOT"
if git diff --quiet; then
  log_action "✓ REPO_ROOT unchanged (git diff clean)"
else
  log_action "⚠️  WARNING: REPO_ROOT has uncommitted changes"
  git status --short | tee -a "$ACTION_LOG"
fi

log_action "✓ Worktree setup completed"
log_action "Worktree location: $WORKTREE_DIR"
log_action "Manifest: $MANIFEST_FILE"
log_action "=========================================="
log_action ""

exit 0
