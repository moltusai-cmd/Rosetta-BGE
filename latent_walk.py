import os
import torch
import sentencepiece as spm
from model import DiffusionRosetta
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def latent_walk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
    # Config Monster 70M
    model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded Monster brain: {ckpt_path}")
    else:
        print(f"❌ Checkpoint introuvable.")
        return

    model.eval()

    # Point de départ (La phrase Meta)
    start_text = "do you’ve like having a split now, Do I’ve now?"
    print(f"\n🚶 STARTING WALK FROM: '{start_text}'")
    print("-" * 50)

    with torch.no_grad():
        emb = encoder.encode([start_text], convert_to_tensor=True, normalize_embeddings=True)
        
        step_size = 0.05 # L'intensité de la dérive
        for i in range(20):
            # On ajoute un peu de "bruit de marche"
            noise = torch.randn_like(emb)
            emb = emb + (step_size * noise)
            emb = F.normalize(emb, p=2, dim=1) # On reste sur la sphère
            
            # Décodage (Temp 0 + Repetition Penalty)
            tokens = torch.full((1, model.num_tokens), model.mask_id, dtype=torch.long, device=device)
            repetition_penalty = 2.0
            
            for t in range(24):
                logits = model(emb, tokens)
                if t > 0:
                    for b in range(1):
                        for token_id in set(tokens[b].tolist()):
                            if token_id != model.mask_id:
                                mask_pos = logits[b, :, token_id] > 0
                                logits[b, mask_pos, token_id] /= repetition_penalty
                                logits[b, ~mask_pos, token_id] *= repetition_penalty
                
                _, predictions = torch.max(logits, dim=-1)
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            result = sp.decode(tokens[0].tolist())
            print(f"Step {i+1:02d} | ✨ {result}")

if __name__ == "__main__":
    latent_walk()
