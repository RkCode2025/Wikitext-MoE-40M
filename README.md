# Wikitext-MoE-40M

This Model is a compact Transformer decoder implementing a Sparse Mixture of Experts (MoE) architecture. It is trained on 30% of Wikitext-103 dataset, a text corpus of Wikipedia articles. It is able to achieve reletively good results while maintaining a low parameter count.

The model was trained on a 30% subset of the WikiText-103 dataset. Evaluation was performed on the WikiText-2 dataset (2 million tokens), yielding 38 ppl and 3.6 cross Entropy

## Key Features

1. Sparse Mixture of Experts (MoE): Implements a routing mechanism with 6 experts and top-k (k=2) selection to optimize compute efficiency.
2. Rotary Positional Embeddings (RoPE): Utilizes relative position encoding for improved long-range dependency handling.
3. Rotary Causal Attention: Integrates RoPE directly into the Multi-Head Attention mechanism for enhanced spatial awareness.
4. RMSNorm & Stability: Employs Root Mean Square Layer Normalization with an epsilon of $1e^{-6}$ for stable training dynamics.
---
## 🛠️ Installation & Setup

To replicate the results of the **Wikitext-MoE-40M** model, follow these steps to clone the repository and set up the environment.

### 1. Clone the Repository
Open your terminal and run the following commands to download the code and enter the project directory:

```bash
git clone [https://github.com/RkCode2025/Wikitext-MoE-40M.git]https://github.com/RkCode2025/Wikitext-MoE-40M.git)
cd Wikitext-MoE-40M
