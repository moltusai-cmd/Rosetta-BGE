import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
from model import DiffusionRosetta
import argparse

def main():
    parser = argparse.ArgumentParser(description="Rosetta Diffusion Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to encode then decode")
    parser.add_argument("--ckpt", type=str, default="rosetta_v5.pt", help="Path to Rosetta weights")
    parser.add_argument("--sp", type=str, default="tokenizer.model", help="Path to SentencePiece model")
    parser.add_argument("--steps", type=int, default=16, help="Number of diffusion steps (12-64 recommended)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.sp)
    
    # 2. Load BGE Encoder (The "Sense")
    print("🛰️ Loading BGE Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # 3. Load Rosetta (The "Mouth")
    print("👄 Loading Rosetta Diffusion...")
    model = DiffusionRosetta(vocab_size=sp.get_piece_size()).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 4. Inference
    print(f"\n📝 Original : {args.text}")
    
    # A. Encode to latent
    with torch.no_grad():
        bge_emb = encoder.encode([args.text], convert_to_tensor=True, normalize_embeddings=True)
    
    # B. Decode back to text via Diffusion
    print(f"🌀 Crystallizing tokens ({args.steps} steps)...")
    gen_ids = model.decode(bge_emb, num_steps=args.steps)
    decoded_text = sp.decode(gen_ids[0].tolist())
    
    print(f"✨ Rosetta  : {decoded_text}")

if __name__ == "__main__":
    main()
