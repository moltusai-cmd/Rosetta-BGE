import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from archive.old_scripts.model_t5 import RosettaT5
from transformers.modeling_outputs import BaseModelOutput

def test_surgical_larynx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Testing Surgical Rosetta-T5 (16 Tokens) on {device}...")

    # 1. Chargement
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5().to(device)
    
    # Correction pour le préfixe _orig_mod (torch.compile)
    state_dict = torch.load("rosetta_t5_surgical_final.pt", map_location=device, weights_only=False)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    # 2. Fragments de test (environ 16 tokens chacun)
    test_fragments = [
        "The small kitten is playing with a red ball",
        "Paris is the beautiful capital city of France",
        "Artificial intelligence will change how we work today",
        "Quantum physics is the study of very small atoms",
        "A quiet forest with tall trees and green grass"
    ]

    print("\n--- Surgical 16-Token Decoding (Limit: 10-20 tokens) ---\n")

    for frag in test_fragments:
        with torch.no_grad():
            # Encodage BGE du fragment
            bge_emb = encoder.encode([frag], convert_to_tensor=True, normalize_embeddings=True)
            bge_emb = bge_emb.detach().clone().to(device).float()

            # Préparation du conditionnement
            hidden_states = model.bge_projector(bge_emb).view(1, 4, model.d_model)
            encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)

            # Génération avec contraintes chirurgicales
            output_ids = model.t5.generate(
                encoder_outputs=encoder_outputs,
                min_length=12,      # On force un peu au dessus de 10 pour la fluidité
                max_length=22,      # On laisse un peu de marge pour l'EOS
                num_beams=4,
                repetition_penalty=2.5,
                early_stopping=True
            )
            
            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            token_count = len(output_ids[0])

            print(f"Original : {frag}")
            print(f"Rosetta  : {decoded_text}")
            print(f"Tokens   : {token_count}")
            print("-" * 30)

if __name__ == "__main__":
    test_surgical_larynx()
