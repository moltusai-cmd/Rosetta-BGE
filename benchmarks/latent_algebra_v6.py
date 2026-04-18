import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from core.model_v6 import RosettaV6
from transformers.modeling_outputs import BaseModelOutput

def decode_vector(model, tokenizer, vector, device):
    with torch.no_grad():
        # Utilisation directe du projecteur de la V6
        x = model.input_norm(vector)
        hidden_states = model.projector(x).view(1, 16, model.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)
        output_ids = model.t5.generate(
            encoder_outputs=encoder_outputs,
            min_length=5,
            max_length=20,
            num_beams=5,
            repetition_penalty=2.5,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def latent_lab_v6():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Latent Algebra Lab V6 | Master Larynx | Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = RosettaV6(num_guides=16).to(device)
    
    state_dict = torch.load('checkpoints/rosetta_v6_epoch_25.pt', map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    encoder.half()

    def get_emb(text):
        emb = encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        return emb.detach().clone().to(device).float()

    experiments = [
        {"name": "Geography", "formula": "Paris - France + Japan", "vec": get_emb("Paris") - get_emb("France") + get_emb("Japan")},
        {"name": "Royalty", "formula": "King - Man + Woman", "vec": get_emb("King") - get_emb("Man") + get_emb("Woman")},
        {"name": "Action", "formula": "Walking - legs + wheels", "vec": get_emb("Walking") - get_emb("legs") + get_emb("wheels")},
        {"name": "Concept Blend", "formula": "Forest + Fire", "vec": (get_emb("Forest") + get_emb("Fire")) / 2}
    ]

    print("\n--- Rosetta-V6 Algebraic Decoding ---\n")
    for exp in experiments:
        vec = F.normalize(exp["vec"], p=2, dim=1)
        result = decode_vector(model, tokenizer, vec, device)
        print(f"🧪 {exp['name']} ({exp['formula']})")
        print(f"🗣️  {result}\n" + "-"*40)

if __name__ == "__main__":
    latent_lab_v6()
