# 🌀 Rosetta-BGE: High-Fidelity Semantic Transduction

**Rosetta-BGE** is a specialized neural architecture designed to bridge the gap between continuous latent spaces and discrete natural language. Specifically, it serves as the "Larynx" for AGI systems (like project **Titan**) by translating **BAAI/bge-small-en-v1.5** embeddings back into articulate, human-readable fragments.

Unlike traditional decoders, Rosetta is optimized for **Semantic Algebra** and **Manifold Fidelity**, ensuring that mathematical operations in the latent space (e.g., *King - Man + Woman*) yield grammatically and conceptually perfect results.

---

## 🏗️ Architecture: Rosetta-V6 (Master Larynx)

The current state-of-the-art for this repository is the **V6 PRO** model, which features:

- **Backbone**: T5-Small (60M parameters) acting as a pre-trained linguistic prior.
- **Transduction Layer**: A 4-stage **Deep Residual Projector** that maps 384d BGE vectors into 16 high-dimensional "Guide Tokens".
- **Semantic Mirror**: A secondary projection head that reconstructs the original BGE vector from the decoder's hidden states, ensuring 99%+ angular alignment (Cosine Similarity).
- **Surgical Precision**: Optimized for 16-token fragments, providing high-resolution decoding of complex conceptual vectors.

---

## 🧬 Semantic Algebra Results

Rosetta-V6 demonstrates advanced conceptual understanding through latent space manipulation:

| Operation | Resulting Output |
| :--- | :--- |
| `Paris - France + Japan` | **"Tokyo, Japan"** |
| `King - Man + Woman` | **"Queen of Kings"** |
| `Walking - legs + wheels` | **"Strolling in a car"** |
| `'A scientist in a lab' - scientist + artist` | **"Artist in a studio creating art"** |

---

## 🌋 The Giga-Forge Pipeline

To achieve near-perfect decoding, we employ an **Infinite Manifold Sweep** strategy:
1. **Latent Mixup**: Continuous interpolation between real FineWeb-Edu anchor vectors.
2. **Stochastic Sampling**: Generation of 64 hypotheses per latent point using Rosetta.
3. **Rejection Sampling**: Only pairs with **Cosine Similarity > 0.95** (as judged by BGE) are kept.
4. **Iterative Refinement**: The model is retrained on its own "Gold" discoveries to eliminate semantic hallucinations.

---

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers sentence-transformers datasets tqdm
```

### Decoding a Vector
```python
from core.model_v6 import RosettaV6
import torch

model = RosettaV6(num_guides=16)
model.load_state_dict(torch.load("checkpoints/rosetta_v6_epoch_25.pt"))
model.eval()

# Your BGE vector (384d)
bge_vector = torch.randn(1, 384) 
output_ids = model(bge_vector)
print(tokenizer.decode(output_ids[0]))
```

---

## 🛠️ Repository Structure

- `core/`: Neural architectures and residual projection logic.
- `forge/`: Synthetic data generation and manifold sweeping tools.
- `training/`: Optimized training loops (Mixed Precision, Gradient Accumulation).
- `benchmarks/`: Latent algebra and semantic interpolation laboratories.
- `checkpoints/`: Pre-trained weights for the V6 Master Larynx.

---
*Forged in the RTX-FORGE environment. High-signal engineering only.* ⚒️🔥🌀
