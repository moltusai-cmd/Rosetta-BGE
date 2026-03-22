import os
import torch
import sentencepiece as spm
from model import DiffusionRosetta
import torch.nn.functional as F

def latent_dark_side():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    
    # Config Monster 70M
    model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded Monster brain: {ckpt_path} (Step {ckpt.get('step', 'unknown')})")
    else:
        print(f"❌ Checkpoint {ckpt_path} introuvable.")
        return

    model.eval()

    print("\n🕵️ EXPLORATION DE LA FACE OBSCURE (100 Random Latent Vectors)\n" + "-"*50)

    with torch.no_grad():
        # Génération de 100 vecteurs aléatoires sur la sphère BGE (384d)
        random_latents = torch.randn(100, 384, device=device)
        random_latents = F.normalize(random_latents, p=2, dim=1) # Normalisation cruciale
        
        for i in range(100):
            emb = random_latents[i:i+1]
            
            # Décodage (on utilise les paramètres optimaux trouvés aujourd'hui)
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
                conf_probs = torch.softmax(logits, dim=-1)
                confidence, _ = torch.max(conf_probs, dim=-1)
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(confidence, num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            result = sp.decode(tokens[0].tolist())
            print(f"[{i+1:03d}] ✨ {result}")

if __name__ == "__main__":
    latent_dark_side()
