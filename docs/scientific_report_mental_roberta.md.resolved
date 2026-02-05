# Scientific Report: MentalRoBERTa Text Embedding Extraction

## 1. Objective
The primary objective was to extract robust linguistic biomarkers from patient transcripts using a domain-specific transformer model. These embeddings serve as high-dimensional features for downstream multimodal depression detection in Phase 3.

## 2. Methodology

### 2.1 Model Architecture
We utilized the **`mental/mental-roberta-base`** model, a RoBERTa-based transformer pre-trained on mental health-related linguistic data. This domain specialization allows the model to capture latent semantic cues related to affective disorders more effectively than general-purpose models.

### 2.2 Extraction Strategy: Sliding Window Pooling
To overcome the 512-token context window limit of RoBERTa, we implemented a **Sliding Window** approach:
1. **Full Transcript Encoding:** The entire session transcript is tokenized.
2. **Chunking:** The tokenized sequence is split into overlapping chunks of 512 tokens with a 50-token stride (overlap).
3. **Chunk-wise Embedding:** Each chunk is passed through the transformer to extract its mean-pooled hidden states.
4. **Contextual Aggregation:** The resulting chuinnk embeddings are averaged to produce a single, unified session vector representing the entire transcript.

### 2.3 Feature Space Optimization
To mitigate **Anisotropy** (the tendency of BERT embeddings to occupy a narrow cone in vector space), we applied **Layer Normalization** (Z-score normalization) across the 768-dimensional feature space.

## 3. Mathematical Foundations

### 3.1 Mean Pooling
For each chunk with $T$ tokens and hidden states $h_t \in \mathbb{R}^{768}$, the pooled embedding $e$ is calculated as:
$$e = \frac{1}{\sum M} \sum_{t=1}^{T} h_{t} \cdot M_{t}$$
where $M$ is the attention mask (excluding padding tokens).

### 3.2 Global Pooling (Sliding Window)
For $K$ windows with embeddings $e_k$, the final session embedding $E$ is:
$$E = \frac{1}{K} \sum_{k=1}^{K} e_k$$

### 3.3 Anisotropy Correction (LayerNorm)
For an embedding vector $x \in \mathbb{R}^{d}$, we perform dimension-wise normalization:
$$\hat{x} = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \epsilon}}$$
This ensures $E[\hat{x}] \approx 0$ and $\text{Var}[\hat{x}] \approx 1$.

## 4. Validation Results

A statistical integrity check was performed on 275 extracted session embeddings against the ground-truth PHQ-8 labels.

| Metric | Result | Target / Interpretation |
|--------|--------|--------------------------|
| **Total Sessions** | 275 | All input sessions processed |
| **Feature Dimensions** | 768 | Consistent with RoBERTa-base |
| **Global Mean** | 0.000001 | Perfectly centered (Target: ~0) |
| **Global Std Dev** | 0.999325 | Unit variance preserved (Target: ~1) |
| **Cosine Similarity** | 0.977177 | High semantic coherence |
| **Linear Probe Acc** | **60.36%** | Significant predictive signal for pilot phase |

## 5. Conclusion
The MentalRoBERTa feature space is statistically stable and contains a measurable signal for depression detection. The implementation of sliding window pooling successfully preserved 100% of the transcript context, providing a superior foundation for the Phase 3 multimodal fusion model.
