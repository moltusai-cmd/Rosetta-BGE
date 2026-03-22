import os
import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
from model import DiffusionRosetta

def algebra_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
    # On utilise le modèle Rosetta-PRO 70M
    model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    
    # On pointe vers le dernier checkpoint du monstre
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ Loaded Monster brain: {ckpt_path}")
    else:
        print(f"⚠️ {ckpt_path} non trouvé, utilisation de rosetta_v5.pt (fallback)")
        ckpt = torch.load("rosetta_v5.pt", map_location=device, weights_only=False)
        # Fallback v5 might need different config (d_model=512)
        model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=512, num_cycles=0).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    def run_equation(name, text_base, text_minus=None, text_plus=None, weight=1.0):
        print(f"\n🧪 Test: {name}")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            result_emb = emb_base
            
            print(f"   Base: '{text_base}'")
            
            if text_minus:
                emb_minus = encoder.encode([text_minus], convert_to_tensor=True, normalize_embeddings=True)
                result_emb = result_emb - (weight * emb_minus)
                print(f"   [-] : '{text_minus}'")
                
            if text_plus:
                emb_plus = encoder.encode([text_plus], convert_to_tensor=True, normalize_embeddings=True)
                result_emb = result_emb + (weight * emb_plus)
                print(f"   [+] : '{text_plus}'")
            
            # Re-normalisation (Crucial pour BGE)
            result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
            
            # Décodage Greedy (temp 0) avec Pénalité de Répétition
            tokens = torch.full((1, model.num_tokens), model.mask_id, dtype=torch.long, device=device)
            repetition_penalty = 1.5
            
            print("\n   🔍 [Top-5 Probabilities Analysis]")
            for t in range(24):
                logits = model(result_emb, tokens)
                
                # Analyse du premier token au début
                if t == 0:
                    probs_analysis = torch.softmax(logits[0, 0], dim=-1)
                    top_p_val, top_idx = torch.topk(probs_analysis, 5)
                    for i in range(5):
                        word = sp.decode([top_idx[i].item()])
                        print(f"      Token #0: '{word}' ({top_p_val[i].item():.2%})")
                
                # Appliquer la pénalité de répétition sur les logits
                if t > 0:
                    for b in range(1):
                        for token_id in set(tokens[b].tolist()):
                            if token_id != model.mask_id:
                                # On applique la pénalité de manière vectorisée sur toute la séquence
                                mask_pos = logits[b, :, token_id] > 0
                                logits[b, mask_pos, token_id] /= repetition_penalty
                                logits[b, ~mask_pos, token_id] *= repetition_penalty

                # Décodage Greedy (Temp 0)
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
            
            decoded = sp.decode(tokens[0].tolist())
            print(f"\n   ✨ Rosetta Final Result: {decoded}")
            print("-" * 50)

    def run_overdrive(name, text_base, text_target, factor=1.5):
        print(f"\n🚀 OVERDRIVE (Extrapolation {factor*100:.0f}%): {name}")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_target = encoder.encode([text_target], convert_to_tensor=True, normalize_embeddings=True)
            
            # Extrapolation: Base + factor * (Target - Base)
            # Ce qui revient à : (1 - factor) * Base + factor * Target
            result_emb = (1 - factor) * emb_base + factor * emb_target
            result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
            
            # Décodage avec Pénalité 2.0
            tokens = torch.full((1, model.num_tokens), model.mask_id, dtype=torch.long, device=device)
            repetition_penalty = 2.0
            
            for t in range(24):
                logits = model(result_emb, tokens)
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
            
            decoded = sp.decode(tokens[0].tolist())
            print(f"   Base  : '{text_base}'")
            print(f"   Target: '{text_target}'")
            print(f"   ✨ Result: {decoded}")
            print("-" * 50)

    # --- TESTS ---
    
    # 1. Overdrive: Nature -> Cyberpunk (150%)
    run_overdrive("Nature vers Cyber-Extrême", 
                  "a peaceful green forest with tall trees and singing birds under a bright sun",
                  "a neon futuristic city with glowing lights and flying cars at night",
                  factor=1.5)

    # 2. Overdrive: Normal -> Amour Absolu (150%)
    run_overdrive("Normal vers Amour-Extrême",
                  "a man is walking down the street alone during the afternoon",
                  "two lovers are hugging each other with deep passion and eternal love forever",
                  factor=1.5)

    # 3. Overdrive: Petit -> Colossal (150%)
    run_overdrive("Petit vers Gigantisme",
                  "a small ant is carrying a tiny piece of bread on the grass",
                  "a massive dragon is burning a huge mountain with blue cosmic fire",
                  factor=1.5)

if __name__ == "__main__":
    algebra_test()

