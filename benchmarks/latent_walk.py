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
        x = model.input_norm(vector)
        hidden_states = model.projector(x).view(1, 16, model.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)
        output_ids = model.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=20,
            num_beams=5,
            repetition_penalty=2.5,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def latent_random_walk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌌 Latent Discovery Lab | Random Walk | Device: {device}")

    # 1. Chargement des modèles
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = RosettaV6(num_guides=16).to(device)
    
    state_dict = torch.load('checkpoints/rosetta_v6_epoch_25.pt', map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    encoder.half()

    # 2. Point d'ancrage
    start_phrase = "A wise old man sitting in a quiet library reading a book"
    print(f"\n⚓ Anchor Concept: '{start_phrase}'\n")
    
    with torch.no_grad():
        anchor_vec = encoder.encode([start_phrase], convert_to_tensor=True, normalize_embeddings=True)
        anchor_vec = anchor_vec.detach().clone().to(device).float()

    # 3. La Marche Aléatoire (Drift)
    # On va s'éloigner du centre en 5 étapes
    steps = 5
    noise_intensity = 0.15 # Force de la mutation à chaque pas

    current_vec = anchor_vec.clone()

    for step in range(1, steps + 1):
        # A. Ajout de bruit gaussien (Mutation)
        noise = torch.randn_like(current_vec) * noise_intensity
        current_vec = current_vec + noise
        
        # B. Re-projection sur la sphère BGE (Crucial pour que ça reste un "concept" valide)
        current_vec = F.normalize(current_vec, p=2, dim=1)
        
        # C. Mesure de la distance avec l'ancrage
        similarity = F.cosine_similarity(anchor_vec, current_vec).item()
        
        # D. Décodage par Rosetta
        result = decode_vector(model, tokenizer, current_vec, device)
        
        print(f"👣 Step {step} (Similarity: {similarity:.2f})")
        print(f"🗣️  Rosetta: {result}\n" + "-"*40)

if __name__ == "__main__":
    latent_random_walk()
