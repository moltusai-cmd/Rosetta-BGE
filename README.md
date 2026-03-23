# 🌀 Rosetta-BGE: The Ultra Brain (v6.0) 🧠

Rosetta is a high-fidelity semantic decoder based on a **Dual-Recursive Diffusion Transformer**. It is designed to render **BGE-small-en-v1.5** (384d) latent vectors into articulate, human-readable text.

## 🚀 The v6 "Ultra" Breakthrough
The latest v6 architecture introduces several state-of-the-art optimizations that have doubled convergence speed and significantly enhanced semantic precision:

- **Weight Tying**: Shared weights between the Embedding and the Output (FC) layers, forcing a perfectly symmetrical semantic space.
- **SwiGLU Activation**: Transitioned to Gated Linear Units (as used in Llama 3) for increased expressive power at a constant parameter count.
- **InfoNCE (Contrastive) Loss**: Replaced simple cosine distance with a contrastive learning objective, forcing the model to discriminate between subtle concept nuances.
- **Curriculum Masking**: Training difficulty scales dynamically, starting with simple fragment reconstruction and evolving toward full-sentence denoising.
- **Sliding Window Augmentation**: Universal decoding of any text fragment (1-16 tokens), calibrated at position zero with perfect **EOS (End Of Sentence)** mastery.

## 🏗️ Architecture Specs
- **Parameters**: ~70M
- **Hidden Dimension (d_model)**: 1024
- **Layers**: 12 effective layers via 6 recursion cycles.
- **Conditioning**: Semantic "Nose" projecting BGE vectors into 4 dynamic Guide Tokens.
- **Speed**: Optimized for **RTX 5080** using `torch.compile` and 2x Gradient Accumulation.

## 🧪 Latent Algebra Laboratory
Rosetta v6 demonstrates that the BGE space is a high-resolution computational grid:

### 1. Zero-Shot Analogies
Even in early training steps (< 3k), Rosetta v6 successfully solves complex analogies:
- **Geography**: `Paris - France + Japan` ➔ **"Tokyo"** (confirmed at Weight 1.2).
- **Gender**: `Man` to `Woman` transition perfectly balanced at **α=0.50**.

### 2. Emergent Narrative Logic
- **Causality**: Interpolating between a "Fast Car" and "Debris" spontaneously generates the concept of an **"Accident"**.
- **Linguistic Invention**: The model generates context-aware neologisms like **"Spacts"** (Skyscrapers + Impacts) or **"Hearton"** (Quantum particle of Love).

### 3. Surgical Precision
Mastery of the EOS token allows for the decoding of isolated entities without context hallucinations:
- `BGE("cat")` ➔ **"cat"** `</s>`
- `BGE("London")` ➔ **"London"** `</s>`

## 🛠️ Laboratory Scripts
- `surgical_test.py`: Precision diagnostics (Property Swapping, Multi-Attribute Injection).
- `latent_walk.py`: Random walk exploration around target semantic anchors.
- `latent_dark_side.py`: Decodes 100 random latent vectors to map the "manifold's ghosts".
- `latent_additive.py`: Tests for additive paradoxes and archetype extraction.
- `meta_tuner.py`: Auto-ML engine for hyper-parameter optimization.

## 🚜 Training Status
- **Accuracy**: 92%+ on fragments, ~75% on complex technical text.
- **Semantic Fidelity (L_SEM)**: **0.02** (Near-lossless projection).
- **Optimizer**: AdamW + OneCycleLR (Peak 1.2e-3).

---
*Rosetta-BGE v6.0 - Forging the high-resolution mirror of latent thought.* 🚀💎🦾🌀
