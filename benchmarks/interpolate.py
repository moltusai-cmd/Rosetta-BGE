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
            max_length=25,
            num_beams=5,
            repetition_penalty=3.0,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def latent_interpolation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌉 Latent Interpolation Lab | Rosetta-V6 | Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = RosettaV6(num_guides=16).to(device)
    state_dict = torch.load('checkpoints/rosetta_v6_epoch_25.pt', map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    encoder.half()

    # Points d'ancrage
    text_a = "A wise old man sitting in a quiet library reading a book"
    text_b = "A massive black hole consuming a bright star in deep space"
    
    print(f"\n🅰️  Start: '{text_a}'")
    print(f"🅱️  End  : '{text_b}'\n")

    with torch.no_grad():
        vec_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True).float()
        vec_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True).float()
        vec_a, vec_b = vec_a.to(device), vec_b.to(device)

    # 8 étapes d'interpolation
    steps = 8
    for i in range(steps + 1):
        alpha = i / steps
        # Interpolation Linéaire (LERP) puis Normalisation (pour rester sur la sphère)
        interp_vec = (1 - alpha) * vec_a + alpha * vec_b
        interp_vec = F.normalize(interp_vec, p=2, dim=1)
        
        res = decode_vector(model, tokenizer, interp_vec, device)
        print(f"🎬 Step {i} (Alpha: {alpha:.2f})")
        print(f"🗣️  Rosetta: {res}\n" + "-"*50)

if __name__ == "__main__":
    latent_interpolation()
