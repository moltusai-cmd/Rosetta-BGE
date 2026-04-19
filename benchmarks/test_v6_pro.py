import torch
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
import sys
import os

# Configuration des imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v6_pro import RosettaV6Pro

def test_v6_pro_master():
    device = torch.device("cuda")
    print(f"💎 Testing Rosetta-V6 PRO MASTER | Device: {device}...")

    # 1. Chargement
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaV6Pro(num_guides=16).to(device)
    
    # Chargement du poids maître (le fruit de tes 30 époques)
    checkpoint = 'checkpoints/rosetta_v6_pro_master.pt'
    if not os.path.exists(checkpoint):
        print(f"❌ Error: {checkpoint} not found!")
        return
        
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    # Nettoyage des clés torch.compile
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    print("🛰️  Encoding test fragments...")
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

    print("\n" + "="*50)
    print("      🏆 ROSETTA V6 PRO MASTER DECODING 🏆")
    print("="*50 + "\n")

    for frag in test_fragments:
        with torch.no_grad():
            # Encodage BGE
            bge_emb = encoder.encode([frag], convert_to_tensor=True, normalize_embeddings=True)
            bge_emb = bge_emb.detach().clone().to(device).float()

            # Décodage Master
            output_ids = model(bge_emb)
            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"📥 SOURCE: {frag}")
            print(f"📤 ROSETTA: {decoded_text}")
            print("-" * 50)

if __name__ == "__main__":
    test_v6_pro_master()
