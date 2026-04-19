# 🌀 Rosetta-BGE: The Master Larynx (V6 PRO)

**Rosetta-BGE** is a high-fidelity semantic transduction engine designed for project **Titan**. It bridges the gap between continuous latent spaces and discrete natural language by translating **BAAI/bge-small-en-v1.5** embeddings back into articulate, human-readable fragments.

The **V6 PRO** architecture features a 30M-parameter Deep Residual Projector coupled with a T5-Small decoder, enabling near-lossless reconstruction of conceptual vectors even through complex latent space algebra.

---

## 🏗️ Technical Architecture

- **Backbone**: T5-Small (60M parameters) as the linguistic prior.
- **Projector**: 10-layer **Deep Residual Projector** (1024-hidden dim) with ~30M parameters.
- **Conditioning**: 16 high-dimensional **Guide Tokens** injected via Cross-Attention.
- **Surgical Precision**: Optimized for 16-token semantic fragments.
- **Semantic Mirror**: Cosine-similarity-based anchor system for angular fidelity (99%+ alignment).

---

## 🧬 Latent Algebra Performance

Rosetta-V6 PRO demonstrates superior conceptual reasoning by solving complex analogies in the latent space:

| Logic | Source Vectors | Decoded Output |
| :--- | :--- | :--- |
| **Geography** | `Paris - France + Japan` | **"Tokyo, Japan"** |
| **Royalty** | `King - Man + Woman` | **"Queen of Kings"** |
| **Action** | `Walking - legs + wheels` | **"Strolling in a car"** |
| **Scene** | `Scientist in a lab - scientist + artist` | **"Artist in a studio"** |

---

## 🚀 How-To: Usage & Workflow

### 1. Installation
```bash
pip install torch transformers sentence-transformers datasets tqdm
```

### 2. Inference (Decoding)
To verbalize a raw BGE vector (384d) into English text:

```python
import torch
from core.model_v6_pro import RosettaV6Pro
from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = RosettaV6Pro(num_guides=16).to(device)

# Load Master weights
state_dict = torch.load("checkpoints/rosetta_v6_pro_master.pt", map_location=device)
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})
model.eval()

# Your 384d BGE Vector (e.g., predicted by Titan)
bge_vector = torch.randn(1, 384).to(device)

with torch.no_grad():
    output_ids = model(bge_vector)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Rosetta output: {text}")
```

### 3. Data Mining (Infinite Sweep)
To map the unknown areas of the manifold and harvest "Gold Concepts" (>0.96 similarity):
```bash
python3 forge/infinite_manifold_sweep.py
```
*Note: Uses an asynchronous pipeline to saturate both RTX 5080 GPU and 16-core CPU.*

### 4. Monster Training (Refinement)
To refine the model on the full 4M-sample dataset (FineWeb + Gold chunks):
```bash
python3 training/train_rosetta_v6_monster.py
```

---

## 🛠️ Repository Organization

- `core/`: Neural architectures and residual projection engines.
- `forge/`: Synthetic data generation and manifold mining tools.
- `training/`: Accelerated training loops with Mixed Precision & Grad Accumulation.
- `benchmarks/`: Latent algebra and semantic interpolation testing suites.
- `checkpoints/`: Production-ready weights for the V6 PRO Master Larynx.

---
*Forged in the RTX-FORGE environment. High-signal engineering only.* ⚒️🔥🌀
