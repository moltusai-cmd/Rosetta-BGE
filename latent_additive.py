import os
import torch
import sentencepiece as spm
from model import DiffusionRosetta
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def latent_additive():
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

    def run_add(name, texts):
        print(f"\n🧪 ADDITIVE SUM: {name} ({' + '.join(texts)})")
        with torch.no_grad():
            # On encode chaque texte
            embs = [encoder.encode([t], convert_to_tensor=True, normalize_embeddings=True) for t in texts]
            
            # SOMME BRUTE (non normalisée d'abord)
            raw_sum = torch.stack(embs).sum(dim=0)
            
            # RE-NORMALISATION (La projection sur la sphère)
            result_emb = F.normalize(raw_sum, p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    def run_sweep(name, text_a, text_b):
        print(f"\n📊 LERP SWEEP: {name} ({text_a} -> {text_b})")
        # On glisse doucement de A vers B
        steps = 11
        with torch.no_grad():
            emb_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True)
            emb_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True)
            
            for i in range(steps):
                w = i / (steps - 1)
                # LERP Sphérique (Moyenne pondérée re-normalisée)
                result_emb = F.normalize((1 - w) * emb_a + w * emb_b, p=2, dim=1)
                
                # Décodage
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
                    num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                    if num_to_mask > 0:
                        _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                        tokens = predictions.clone()
                        tokens[0, mask_indices] = model.mask_id
                    else:
                        tokens = predictions
                
                print(f"   [w={w:.1f}] ✨ {sp.decode(tokens[0].tolist())}")

    print("\n--- 🌊 NARRATIVE BLENDING (16 Tokens AVG) ---")
    
    def run_narrative_avg(name, text_a, text_b):
        print(f"\n🎭 BLENDING: {name}")
        # On regarde le point de départ, le point d'arrivée et surtout le MILIEU (0.5)
        steps = [0.0, 0.5, 1.0]
        with torch.no_grad():
            emb_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True)
            emb_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True)
            
            for w in steps:
                result_emb = F.normalize((1 - w) * emb_a + w * emb_b, p=2, dim=1)
                
                # Décodage
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
                    num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                    if num_to_mask > 0:
                        _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                        tokens = predictions.clone()
                        tokens[0, mask_indices] = model.mask_id
                    else:
                        tokens = predictions
                
                print(f"   [w={w:.1f}] ✨ {sp.decode(tokens[0].tolist())}")

    run_narrative_avg("Désert & Océan", 
                      "a hot dry desert with yellow sand dunes and a cactus under a burning sun",
                      "a cold deep ocean with blue water waves and a whale under a grey sky")
    
    run_narrative_avg("Cité & Forêt",
                      "a futuristic city with neon lights and tall metal skyscrapers and flying cars at night",
                      "an ancient forest with giant green trees and wild animals and waterfalls today")
    
    run_narrative_avg("Silence & Bruit",
                      "a quiet library with wooden shelves and old books and people studying in silence",
                      "a loud rock concert with electric guitars and flashing lights and people dancing fast")

    print("\n--- ✂️ NARRATIVE SUBTRACTION (16 Tokens SUB) ---")

    def run_narrative_sub(name, text_base, text_sub, weight=1.0):
        print(f"\n✂️ SUBTRACTION: {name} ({text_base} MINUS '{text_sub}')")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_sub = encoder.encode([text_sub], convert_to_tensor=True, normalize_embeddings=True)
            
            # On soustrait et on re-normalise
            result_emb = F.normalize(emb_base - (weight * emb_sub), p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    run_narrative_sub("Ville sans Bâtiments", 
                      "a busy city with tall buildings and crowded streets", 
                      "tall buildings")
    
    run_narrative_sub("Forêt sans Arbres", 
                      "a green forest with giant trees and wild animals", 
                      "giant trees")
    
    print("\n--- 🏛️ ARCHEtype EXTRACTION (Multi-AVG) ---")

    def run_archetype(name, texts):
        print(f"\n🏛️ ARCHETYPE: {name}")
        with torch.no_grad():
            embs = [encoder.encode([t], convert_to_tensor=True, normalize_embeddings=True) for t in texts]
            # Moyenne brute de tous les vecteurs
            archetype_emb = torch.stack(embs).mean(dim=0)
            # Re-normalisation sur la sphère BGE
            result_emb = F.normalize(archetype_emb, p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    run_archetype("La Connaissance", [
        "an old professor teaching students in a large university",
        "a complex scientific book with many pages and mathematical formulas",
        "a glowing brain with neural connections and blue digital light",
        "a small child reading a fairy tale under a lamp at night"
    ])

    run_archetype("Le Danger", [
        "a hungry grey wolf showing teeth in the dark forest",
        "a dangerous bomb with a ticking timer and red wires",
        "a massive tornado destroying houses and trees during a storm",
        "a dark narrow street with a mysterious man hiding in shadows"
    ])

    run_archetype("La Paix", [
        "a calm blue lake reflecting the white mountains in summer",
        "a peaceful baby sleeping in a white cradle with a smile",
        "a Buddhist monk meditating in silence inside a golden temple",
        "a beautiful orange sunset over the quiet ocean waves today"
    ])

    print("\n--- 👻 CHIRURGIE DE L'ABSENCE (Retirer l'Humain) ---")

    def run_absence(name, text_base, text_human):
        print(f"\n👻 ABSENCE: {name} ({text_base} MINUS '{text_human}')")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_human = encoder.encode([text_human], convert_to_tensor=True, normalize_embeddings=True)
            
            # Soustraction renforcée (1.2) pour bien effacer l'humain
            result_emb = F.normalize(emb_base - (1.2 * emb_human), p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    run_absence("Café Vide", "a crowded cafe with people talking and laughing", "people")
    run_absence("Guerre sans Hommes", "soldiers fighting with guns in a muddy battlefield", "soldiers")
    run_absence("Science sans Chercheur", "a scientist looking through a microscope in a laboratory", "scientist")
    run_absence("Concert Fantôme", "a guitarist playing electric guitar on a loud stage", "guitarist")

    print("\n--- 🤖 CHIRURGIE PROTHÉTIQUE (Remplacement Latent) ---")

    def run_replacement(name, text_base, text_sub, text_add):
        print(f"\n🔄 REPLACEMENT: {name} ({text_base} - '{text_sub}' + '{text_add}')")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_sub = encoder.encode([text_sub], convert_to_tensor=True, normalize_embeddings=True)
            emb_add = encoder.encode([text_add], convert_to_tensor=True, normalize_embeddings=True)
            
            # Formule: Base - Sub + Add (re-normalisée)
            result_emb = F.normalize(emb_base - emb_sub + emb_add, p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    run_replacement("Café Cyber", "a crowded cafe with people talking and laughing", "people", "robots")
    run_replacement("Guerre de Brume", "soldiers fighting with guns in a muddy battlefield", "soldiers", "thick white fog")
    run_replacement("IA Laborantine", "a scientist looking through a microscope in a laboratory", "scientist", "supercomputer")
    run_replacement("Radio Stage", "a guitarist playing electric guitar on a loud stage", "guitarist", "old radio")

    print("\n--- 🚀 SAUT TECHNOLOGIQUE (Future Replacement) ---")

    def run_future_swap(name, text_base, text_sub, text_add):
        print(f"\n⚡ FUTURE: {name} ({text_base} -> FUTURE)")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_sub = encoder.encode([text_sub], convert_to_tensor=True, normalize_embeddings=True)
            emb_add = encoder.encode([text_add], convert_to_tensor=True, normalize_embeddings=True)
            
            # Formule: Base - Sub + 1.2 * Add
            result_emb = F.normalize(emb_base - emb_sub + (1.2 * emb_add), p=2, dim=1)
            
            # Décodage
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
                num_to_mask = max(0, int(model.num_tokens * (1 - (t + 1) / 24)))
                if num_to_mask > 0:
                    _, mask_indices = torch.topk(torch.softmax(logits, dim=-1).max(dim=-1)[0], num_to_mask, largest=False, dim=-1)
                    tokens = predictions.clone()
                    tokens[0, mask_indices] = model.mask_id
                else:
                    tokens = predictions
            
            print(f"   ✨ Result : {sp.decode(tokens[0].tolist())}")

    run_future_swap("Écriture", "a man writing a letter with a pen on paper", "letter pen paper", "digital neural hologram")
    run_future_swap("Transport", "an old horse pulling a wooden carriage on a road", "horse wooden carriage", "hovering plasma drone")
    run_future_swap("Repas", "a family eating a fresh apple in a kitchen", "fresh apple kitchen", "synthetic nutrient pill laboratory")
    run_future_swap("Sommeil", "a woman sleeping in a soft bed under a blanket", "soft bed blanket", "cryogenic stasis pod fluid")

if __name__ == "__main__":
    latent_additive()
