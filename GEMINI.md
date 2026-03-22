# 🌀 ROSETTA_BGE Instructional Context (GEMINI.md)

This document serves as the foundational mandate for all AI interactions within the `Rosetta_BGE` workspace. It codifies the architectural principles, technical standards, and operational workflows of the Rosetta project.

---

## 🚀 Project Overview
**Rosetta** is a high-fidelity **Recursive Discrete Diffusion** decoder. Its primary purpose is to translate continuous latent vectors (specifically **BAAI/bge-small-en-v1.5**) back into human-readable text. It acts as the "Broca's Area" for AGI systems (like Albert/AXIMILATOR) that reason in conceptual latent spaces rather than discrete tokens.

### Main Technologies
- **Core**: PyTorch, Sentence-Transformers (BGE-small-en-v1.5).
- **Architecture**: Recursive Transformer with weight sharing (ALBERT-style).
- **Training**: Discrete Diffusion (Mask-Predict) with **Semantic Mirror** (BGE-loss).
- **Optimization**: `torch.compile(mode="reduce-overhead")`, BF16 mixed precision.
- **Dataset**: "Holy Trinity" (FineWeb-Edu, WikiText-103, OpenEdu) - 20M+ latent pairs.

---

## 🏗️ Technical Architecture (Rosetta-Pro 70M)
The current gold standard for the project is the **70M Pro** model:
- **Latent Space**: 384D (BGE).
- **Projection**: 384D -> 4 Guide Tokens (Conditioning).
- **Internal Brain**: `d_model=1024`, 16 Heads.
- **Recursion**: 2 unique layers cycled **6 times** (12 logical steps).
- **Semantic Mirror**: A projection head that reconstructs the original BGE vector from Transformer hidden states to ensure semantic consistency.

---

## 🛠️ Building and Running

### 📦 Environment
- Always use the local virtual environment: `./venv/bin/python`.
- Core dependencies: `torch`, `sentence-transformers`, `sentencepiece`, `datasets`, `tqdm`.

### 🚜 Training (Monster Suite)
To train the 70M model on the 20M monster dataset:
```bash
./venv/bin/python train_monster.py --batch-size 256 --epochs 5 --lr 3e-4
```
*Note: Training includes JIT compilation and BF16 by default.*

### 🗣️ Inference & Testing
- **Simple Decode**: `python decode.py --text "your sentence" --steps 24`
- **Latent Algebra**: `python latent_algebra_test.py` (Tests conceptual math like King - Man + Woman).
- **Quality Bench**: `python benchmark.py` (Measures average semantic similarity).

### 📉 Conversion
- Use `convert_fp16.py` to compress models for distribution (~26MB for Mini, ~130MB for Pro).

---

## 🧬 Development Conventions

### 1. The "Mad Scientist" Ethics
- **Efficiency First**: Favor AGI-on-CPU architectures. Keep models small enough for L3 cache but deep enough for reasoning via recursion.
- **Semantic Integrity**: Every change to the decoder must be validated against `latent_algebra_test.py` to ensure it understands *concepts*, not just words.
- **Data Gavage**: When in doubt, more high-quality data (FineWeb-Edu) is the solution.

### 2. Coding Style
- **RMSNorm**: Always prefer `RMSNorm` over `LayerNorm` for stability in recursive architectures.
- **Differentiable Cycles**: When implementing new loss functions, ensure they are differentiable or use "soft-embedding" proxies.
- **FP16/BF16 Awareness**: Be mindful of `dtype` mismatches when moving between datasets (stored in FP16) and models (trained in FP32/BF16).

### 3. File Naming
- `*monster*`: Refers to the 20M+ example training pipeline.
- `*mini*`: Refers to the 13M parameter baseline.
- `*pro*`: Refers to the 70M parameter recursive powerhouse.

---

## 🛰️ Integration with Albert/AXIMILATOR
Rosetta is the "Mouth" for the **Albert** brain. 
- **Albert** (Mamba/MoE) predicts the next 384D BGE vector.
- **Rosetta** verbalizes that vector.
Any update to the latent space mapping must be synchronized with the Albert project specs in `/home/nini/Albert`.

---
*Forged in the RTX-FORGE environment. High-signal engineering only.* ⚒️🔥🌀
