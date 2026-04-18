import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from archive.old_scripts.model_t5 import RosettaT5

def test_larynx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎤 Testing Rosetta-T5 Larynx on {device}...")

    # 1. Chargement des composants
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5().to(device)
    
    # Nettoyage du state_dict pour enlever le préfixe '_orig_mod.' (dû à torch.compile)
    state_dict = torch.load("rosetta_t5_larynx_final.pt", map_location=device, weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    # 2. Phrases de test
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "A small cat is sitting on a wooden table.",
        "The capital of France is Paris.",
        "Quantum physics is a complex subject for many students."
    ]

    print("\n--- Rosetta-T5 Decoding Results ---\n")

    for sentence in test_sentences:
        with torch.no_grad():
            # Encodage BGE
            bge_emb = encoder.encode([sentence], convert_to_tensor=True, normalize_embeddings=True)
            bge_emb = bge_emb.detach().clone().to(device).float()

            # Décodage via T5 Larynx
            output_ids = model(bge_emb)
            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"Original : {sentence}")
            print(f"Rosetta  : {decoded_text}")
            print("-" * 30)

if __name__ == "__main__":
    test_larynx()
