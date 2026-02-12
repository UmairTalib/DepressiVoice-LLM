# Phase 3 (v9) Technical Report
## Bi-directional Co-Attention Foundation

**Date:** February 7, 2026  
**Status:** Completed  
**Subject:** First successful fusion of frozen WavLM and RoBERTa features.

---

### 1. Objective
The primary objective of Phase 3 (v9) was to establish a robust foundation for multimodal fusion in the "DepressiVoice-LLM" framework. This phase focused on integrating high-level paralinguistic features from Audio (WavLM) with semantic context from Text (Mental-RoBERTa) using a bi-directional attention mechanism.

### 2. Methodology: Bi-directional Co-Attention
In v9, we moved beyond simple concatenation to a **Bi-directional Co-Attention** mechanism. This architecture allows each modality to "inform" the other:

*   **Audio-centric Head:** Audio features serve as the Query ($Q$), searching across Text features (Key $K$ and Value $V$) to find relevant semantic context for specific acoustic events (e.g., matching a sigh with the silence or words surrounding it).
*   **Text-centric Head:** Text features serve as the Query ($Q$), searching across Audio features ($K, V$) for prosodic grounding. This helps the model identify when "neutral" words are spoken with a depressive acoustic signature (e.g., flat affect or jitter).

The outputs of these two heads are fused into a shared **256-dimensional space**, creating a unified sequence representation.

### 3. Data Specification & Regularization
Given the relatively small size of the E-DAIC dataset (275 sessions), regularization was a critical design priority:

*   **Frozen Backbones:** Both **WavLM-Large** and **Mental-RoBERTa** backbones are strictly **FROZEN**. No fine-tuning is performed on these massive models to prevent catastrophic overfitting on the limited clinical samples.
*   **Audio Features:** Extracted from WavLM-Large using weighted sums. In v9, we prioritized **Layer 12** to capture high-level paralinguistic and prosodic information while discarding lower-level acoustic phonetics.
*   **Text Features:** Mental-RoBERTa embeddings aligned to the exact 8-second audio windows.
*   **Dropout:** High dropout rates (0.4) were applied within the fusion layers to ensure the model learns robust, non-redundant multimodal patterns.

### 4. Reasoning: 8-Second Windowing
The selection of **8-second windows** as the "local" unit of analysis was based on clinical and technical reasoning:
*   **Prosodic Stability:** 8 seconds is sufficient to capture a complete phrase or several bursts of speech, allowing for the detection of rhythm and pitch trends.
*   **Contextual Alignment:** It provides enough text (approx. 10-30 words) for RoBERTa to extract meaningful semantic "pockets" without being overwhelmed by the noise of an entire 20-minute interview.
*   **Computational Efficiency:** Allows for high-resolution temporal analysis while remaining within the memory constraints of modern GPU architectures during the Phase 4 temporal classification.

### 5. Infrastructure & Environment
*   **Host:** `g755389@omni.zimt.uni-siegen.de`
*   **Directory:** `~/DepressiVoice-LLM/src/phase3_fusion_v11/`
*   **Environment:** `depressivoice-phase1` (Python 3.10)
*   **Hardware:** Optimized for NVIDIA A100 execution on the cluster.

---

### 6. Validation Results
Initial validation confirmed that the shared 256-dimensional representation successfully captures cross-modal signals. The model cleared the "random guess" hurdle (0.50 AUC) early in training, proving that the Bi-directional Co-Attention successfully grounds depressive words in their acoustic reality.

---
*Report generated for the DepressiVoice-LLM Technical Series.*
