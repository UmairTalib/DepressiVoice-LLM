#!/bin/bash
# Quick test script for Phase 2B text extraction
# Tests a single session locally before cluster deployment

set -e

echo "========================================"
echo "Phase 2B Quick Test"
echo "========================================"

# Check if we're on cluster or local
if [ -d "/home/g755389/DepressiVoice-LLM" ]; then
    PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
    echo "Environment: CLUSTER"
else
    PROJECT_ROOT="/Users/umairtalib/DepressiVoice-LLM"
    echo "Environment: LOCAL"
fi

# Configuration
TEST_SESSION="300"
INPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
OUTPUT_DIR="$PROJECT_ROOT/data/phase2_clean/text_embeddings_raw"
SCRIPT="$PROJECT_ROOT/src/phase2_text/extract_embeddings.py"

echo "Test Session: $TEST_SESSION"
echo "Script: $SCRIPT"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import sentence_transformers; import librosa; import torch" 2>/dev/null || {
    echo "✗ Missing dependencies. Install with:"
    echo "  pip install sentence-transformers librosa torch"
    exit 1
}
echo "✓ Dependencies OK"
echo ""

# Run extraction
echo "Running extraction..."
python3 $SCRIPT \
    --session_id "$TEST_SESSION" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --verbose

echo ""
echo "========================================"
echo "Verification"
echo "========================================"

# Check output file
OUTPUT_FILE="$OUTPUT_DIR/${TEST_SESSION}_embeddings.pt"
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file created: $OUTPUT_FILE"
    
    # Check shape
    python3 << EOF
import torch
emb = torch.load("$OUTPUT_FILE")
print(f"Shape: {emb.shape}")
print(f"Dtype: {emb.dtype}")
print(f"Range: [{emb.min():.2f}, {emb.max():.2f}]")
print(f"Mean: {emb.mean():.2f}")

# Validate
assert emb.ndim == 2, "Should be 2D tensor"
assert emb.shape[1] == 768, "Should have 768 dimensions"
assert not torch.isnan(emb).any(), "Should not have NaN"
assert not torch.isinf(emb).any(), "Should not have Inf"
print("✓ All checks passed")
EOF
else
    echo "✗ Output file not found: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo "Ready for cluster deployment!"
