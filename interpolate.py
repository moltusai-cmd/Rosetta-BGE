import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
from model import DiffusionRosetta

def interpolate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    model = DiffusionRosetta(vocab_size=sp.get_piece_size()).to(device)
    ckpt = torch.load("rosetta_v5.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Two distinct concepts
    text_a = "A fast sports car on the highway."
    text_b = "A bicycle on a quiet city street."

    print(f"🅰️ Phrase A : {text_a}")
    print(f"🅱️ Phrase B : {text_b}")
    print("\n🌀 Mixing concepts in latent space...")

    with torch.no_grad():
        # Encode both to BGE latent space
        emb_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True)
        emb_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True)

        # Interpolate at different ratios
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        print(f"\n{'Ratio A/B':<10} | {'Rosetta Interpretation'}")
        print("-" * 60)
        
        for r in ratios:
            # Linear interpolation: v = (1-r)*A + r*B
            mixed_emb = (1 - r) * emb_a + r * emb_b
            # Re-normalize (BGE expects unit vectors)
            mixed_emb = torch.nn.functional.normalize(mixed_emb, p=2, dim=1)
            
            # Decode the hybrid vector
            gen_ids = model.decode(mixed_emb, num_steps=24) # More steps for better quality
            decoded = sp.decode(gen_ids[0].tolist())
            
            label = f"{int((1-r)*100)}% A / {int(r*100)}% B"
            print(f"{label:<10} | {decoded}")

if __name__ == "__main__":
    interpolate()
