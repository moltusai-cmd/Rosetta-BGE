import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from archive.old_scripts.model_t5 import RosettaT5
from transformers.modeling_outputs import BaseModelOutput

def decode_vector(model, tokenizer, vector, device):
    with torch.no_grad():
        # Préparation du conditionnement
        hidden_states = model.bge_projector(vector).view(1, 8, model.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)

        # Génération
        output_ids = model.t5.generate(
            encoder_outputs=encoder_outputs,
            min_length=5,
            max_length=20,
            num_beams=5,
            repetition_penalty=2.5,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def latent_lab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Latent Algebra Lab | Rosetta-T5 PRO | Device: {device}")

    # 1. Chargement
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5(num_guides=8).to(device)
    state_dict = torch.load("rosetta_t5_pro_final.pt", map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    def get_emb(text):
        emb = encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        return emb.detach().clone().to(device).float()

    # 2. Les Expériences
    experiments = [
        {
            "name": "Geography Analogy",
            "formula": "Paris - France + Japan",
            "vec": get_emb("Paris") - get_emb("France") + get_emb("Japan")
        },
        {
            "name": "Royalty Analogy",
            "formula": "King - Man + Woman",
            "vec": get_emb("King") - get_emb("Man") + get_emb("Woman")
        },
        {
            "name": "Action / Object",
            "formula": "Walking - legs + wheels",
            "vec": get_emb("Walking") - get_emb("legs") + get_emb("wheels")
        },
        {
            "name": "Concept Blend",
            "formula": "A forest + Fire",
            "vec": (get_emb("A forest") + get_emb("A fire")) / 2
        }
    ]

    print("\n--- Rosetta-T5 Algebraic Decoding ---\n")

    for exp in experiments:
        # On normalise le vecteur résultant (BGE est un espace sphérique)
        vec = torch.nn.functional.normalize(exp["vec"], p=2, dim=1)
        
        result = decode_vector(model, tokenizer, vec, device)
        
        print(f"🧪 Exp: {exp['name']}")
        print(f"📐 Op : {exp['formula']}")
        print(f"🗣️ Res: {result}")
        print("-" * 40)

if __name__ == "__main__":
    latent_lab()
