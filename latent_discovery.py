import os
import torch
import sentencepiece as spm
from model import DiffusionRosetta
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def latent_discovery():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
    model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded Monster brain: {ckpt_path}")
    
    model.eval()

    def complete_rosetta(prefix_text, context_text):
        print(f"\n📝 Completing: '{prefix_text}' (Context: '{context_text}')")
        with torch.no_grad():
            # 1. On encode le CONTEXTE (l'idée globale)
            emb = encoder.encode([context_text], convert_to_tensor=True, normalize_embeddings=True)
            
            # 2. On prépare les tokens avec le PREFIXE
            prefix_tokens = sp.encode(prefix_text, out_type=int)
            num_prefix = len(prefix_tokens)
            
            # Initialisation : Préfixe + MASKs
            tokens = torch.full((1, model.num_tokens), model.mask_id, dtype=torch.long, device=device)
            tokens[0, :num_prefix] = torch.tensor(prefix_tokens, device=device)
            
            repetition_penalty = 2.0
            for t in range(24):
                logits = model(emb, tokens)
                
                # --- VERROUILLAGE DU PREFIXE ---
                # On ne touche jamais aux tokens du début
                if t > 0:
                    for token_id in set(tokens[0].tolist()):
                        if token_id != model.mask_id:
                            mask_pos = logits[0, :, token_id] > 0
                            logits[0, mask_pos, token_id] /= repetition_penalty
                            logits[0, ~mask_pos, token_id] *= repetition_penalty
                
                _, predictions = torch.max(logits, dim=-1)
                
                # On ré-injecte le préfixe de force dans les prédictions
                predictions[0, :num_prefix] = torch.tensor(prefix_tokens, device=device)
                
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    # On ne masque QUE les tokens après le préfixe
                    conf_probs = torch.softmax(logits, dim=-1)
                    confidence, _ = torch.max(conf_probs, dim=-1)
                    # On met une confiance infinie au préfixe pour ne pas le masquer
                    confidence[0, :num_prefix] = 100.0
                    
                    _, mask_indices = torch.topk(confidence, num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Rosetta completes: {sp.decode(tokens[0].tolist())}")

    # --- INTERROGATION PAR COMPLÉTION ---
    
    # On utilise le vecteur du "Saut Technologique" (Future) comme contexte
    future_context = "a flying plasma remover on the honon"
    
    complete_rosetta("A honon is a", future_context)
    complete_rosetta("The honon is used to", future_context)
    complete_rosetta("A flying plasma remover is", future_context)
    complete_rosetta("To use a plasma remover,", future_context)
    
    # Et pour le Hearton
    complete_rosetta("A hearton is a part of", "quantum physics and love is 42")

if __name__ == "__main__":
    latent_discovery()
