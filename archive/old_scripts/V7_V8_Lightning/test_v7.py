import torch
from sentence_transformers import SentenceTransformer
import sys
import os

# Configuration des imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v7 import RosettaV7

def test_v7_inference(text_input):
    device = torch.device("cpu")
    
    print(f"🛰️  Encoding source: '{text_input}'...")
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    bge_emb = bge_model.encode(text_input, convert_to_tensor=True, normalize_embeddings=True)
    
    print("🧠 Loading Rosetta-V7 Ultra Brain...")
    model = RosettaV7(num_guides=16).to(device)
    
    # On utilise le dernier checkpoint de l'entraînement en cours
    checkpoint = 'checkpoints/rosetta_v7_ultra_latest.pt'
    if os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location=device, weights_only=False)
        # On nettoie les clés du state_dict si compile a été utilisé
        new_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(new_sd)
        model.eval()
        print(f"✅ Loaded checkpoint: {checkpoint}")
    else:
        print("❌ No checkpoint found! Train for at least one epoch.")
        return

    print("👄 Rosetta is thinking and speaking...")
    with torch.no_grad():
        # On passe le vecteur BGE (1, 384)
        output_ids = model(bge_emb.unsqueeze(0).float())
        
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    print("-" * 50)
    print(f"📥 INPUT BGE  : {text_input}")
    print(f"📤 OUTPUT T5 : {result}")
    print("-" * 50)

if __name__ == "__main__":
    # On prend l'argument ou une phrase par défaut
    input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The artificial intelligence is learning to decode the human thoughts from latent space."
    test_v7_inference(input_text)
