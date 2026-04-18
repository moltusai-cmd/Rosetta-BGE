import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from archive.old_scripts.model_t5 import RosettaT5
from transformers.modeling_outputs import BaseModelOutput

def test_larynx_pro():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💎 Testing Rosetta-T5 PRO (8-Guides + Mirror) on {device}...")

    # 1. Chargement (Modèle PRO avec 8 guides)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5(num_guides=8).to(device)
    
    # Nettoyage du state_dict (torch.compile)
    state_dict = torch.load("rosetta_t5_pro_final.pt", map_location=device, weights_only=False)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    # Fragments de test
    test_fragments = [
        "The small kitten is playing with a red ball",
        "Paris is the beautiful capital city of France",
        "Artificial intelligence will change how we work today",
        "Quantum physics is the study of very small atoms",
        "A quiet forest with tall trees and green grass"
    ]

    print("\n--- PRO 8-Token Decoding Results ---\n")

    for frag in test_fragments:
        with torch.no_grad():
            bge_emb = encoder.encode([frag], convert_to_tensor=True, normalize_embeddings=True)
            bge_emb = bge_emb.detach().clone().to(device).float()

            # Décodage PRO
            output_ids = model(bge_emb)
            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"Original : {frag}")
            print(f"Rosetta  : {decoded_text}")
            print("-" * 30)

if __name__ == "__main__":
    test_larynx_pro()
