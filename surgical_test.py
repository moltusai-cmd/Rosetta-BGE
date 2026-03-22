import os
import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
from model import DiffusionRosetta

def surgical_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
    # Config Monster 70M
    model = DiffusionRosetta(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        
        # Nettoyage des clés (remove _orig_mod. from torch.compile)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded Monster brain: {ckpt_path}")
    else:
        print(f"❌ Checkpoint {ckpt_path} introuvable.")
        return

    model.eval()

    def decode_input(text):
        print(f"\n🧪 Testing: '{text}'")
        with torch.no_grad():
            emb = encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            
            # Décodage deterministe (Temp 0)
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
            
            # Analyse brute des IDs pour voir les EOS
            token_ids = tokens[0].tolist()
            decoded_text = sp.decode(token_ids)
            
            # On cherche l'EOS (souvent id=2)
            eos_id = sp.eos_id()
            print(f"   Raw Tokens: {token_ids}")
            print(f"   ✨ Result : {decoded_text}")
            if eos_id in token_ids:
                pos = token_ids.index(eos_id)
                print(f"   ✅ EOS detected at position {pos}!")

    def run_algebra_sweep(name, text_base, text_minus=None, text_plus=None):
        print(f"\n📊 SWEEP: {name} ({text_base} - {text_minus} + {text_plus})")
        weights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_minus = encoder.encode([text_minus], convert_to_tensor=True, normalize_embeddings=True) if text_minus else None
            emb_plus = encoder.encode([text_plus], convert_to_tensor=True, normalize_embeddings=True) if text_plus else None
            
            for w in weights:
                result_emb = emb_base
                if emb_minus is not None:
                    result_emb = result_emb - (w * emb_minus)
                if emb_plus is not None:
                    result_emb = result_emb + (w * emb_plus)
                
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                
                res = sp.decode(tokens[0].tolist())
                print(f"   [W={w:.1f}] ✨ {res}")

    def run_algebra(name, text_base, text_minus=None, text_plus=None, weight=1.0):
        print(f"\n🧪 Algebra: {name} ({text_base} - {text_minus} + {text_plus})")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            result_emb = emb_base
            
            if text_minus:
                emb_minus = encoder.encode([text_minus], convert_to_tensor=True, normalize_embeddings=True)
                result_emb = result_emb - (weight * emb_minus)
            if text_plus:
                emb_plus = encoder.encode([text_plus], convert_to_tensor=True, normalize_embeddings=True)
                result_emb = result_emb + (weight * emb_plus)
            
            result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
            
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

    def run_semantic_op(name, text_a, text_b, op="avg", weight=1.0):
        print(f"\n🧩 OP: {name} ({text_a} {op} {text_b})")
        with torch.no_grad():
            emb_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True)
            emb_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True)
            
            if op == "avg":
                result_emb = (emb_a + emb_b) / 2.0
            elif op == "sub":
                result_emb = emb_a - (weight * emb_b)
            elif op == "add":
                result_emb = emb_a + (weight * emb_b)
                
            result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
            
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

    def run_lerp(name, text_start, text_end, steps=5):
        print(f"\n📊 LERP: {name} ({text_start} -> {text_end})")
        with torch.no_grad():
            emb_start = encoder.encode([text_start], convert_to_tensor=True, normalize_embeddings=True)
            emb_end = encoder.encode([text_end], convert_to_tensor=True, normalize_embeddings=True)
            
            for i in range(steps):
                alpha = i / (steps - 1)
                result_emb = (1 - alpha) * emb_start + alpha * emb_end
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                
                print(f"   [α={alpha:.2f}] ✨ {sp.decode(tokens[0].tolist())}")

    # --- RUN TESTS ---
    print("--- 🎯 CHIRURGIE SIMPLE ---")
    decode_input("cat")
    
    print("\n--- 📊 INTERPOLATION (LERP) ---")
    run_lerp("Mutation Elémentaire", "Sea", "Fire", steps=5)
    run_lerp("Genre Fluide", "Man", "Woman", steps=5)

    print("\n--- 🧩 OPÉRATIONS AFFINÉES (Poids faible) ---")
    run_semantic_op("Chat Électrique", "Cat", "Electricity", op="add", weight=0.6)
    run_semantic_op("Peinture sans Beauté (Atténué)", "Beautiful Painting", "Beauty", op="sub", weight=0.7)

    print("\n--- 🐉 CRÉATION DE CHIMÈRES ---")
    run_semantic_op("Chimère: Dragon d'Eau", "Dragon", "Water", op="avg")
    run_semantic_op("Chimère: Soleil de Nuit", "Sun", "Night", op="avg")
    run_semantic_op("Paradoxe: Lumière Sombre", "Light", "Darkness", op="avg")

    print("\n--- 🐉 HYBRIDATION PONDÉRÉE (Sauver le Dragon) ---")
    # On donne 80% au Dragon pour voir s'il survit à l'eau
    run_semantic_op("Dragon dominant", "Dragon", "Water", op="add", weight=0.2) 
    run_semantic_op("Feu atténué", "Sea", "Fire", op="add", weight=0.3)

    print("\n--- 🥐 SYNTHÈSE D'OBJETS ---")
    run_semantic_op("Synthèse: Petit-Déjeuner", "Coffee", "Ice", op="avg")
    run_semantic_op("Synthèse: Boulangerie", "Bread", "Fire", op="avg")
    run_semantic_op("Synthèse: Météo", "Rain", "Sun", op="avg")
    run_semantic_op("Synthèse: Musique", "Guitar", "Electricity", op="avg")

    print("\n--- 🔄 LATENT PROPERTY SWAPPING ---")
    run_algebra("Swap de Couleur", "A blue flower", "blue", "red")
    run_algebra("Swap de Température", "A hot coffee", "hot", "ice")
    run_algebra("Swap de Capitales", "The capital of France", "France", "Japan")
    run_algebra("Soustraction d'Attribut", "A small house", "small", None)
    run_algebra("Addition d'Attribut", "A house", None, "scary")

    def run_precision_sweep(name, text_base, text_minus=None, text_plus=None):
        print(f"\n🎯 PRECISION SWEEP: {name}")
        weights = [0.6, 0.8, 1.0, 1.2, 1.4]
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_minus = encoder.encode([text_minus], convert_to_tensor=True, normalize_embeddings=True) if text_minus else None
            emb_plus = encoder.encode([text_plus], convert_to_tensor=True, normalize_embeddings=True) if text_plus else None
            
            for w in weights:
                result_emb = emb_base
                if emb_minus is not None:
                    result_emb = result_emb - (w * emb_minus)
                if emb_plus is not None:
                    result_emb = result_emb + (w * emb_plus)
                
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                
                print(f"   [W={w:.1f}] ✨ {sp.decode(tokens[0].tolist())}")

    print("\n--- 🎯 PRECISION SWEEP ---")
    run_precision_sweep("Maison Inquiétante", "A house", None, "scary")
    run_precision_sweep("Café Glacé", "A hot coffee", "hot", "ice")
    run_precision_sweep("Capitale Japon", "The capital of France", "France", "Japan")

    def run_multi_algebra(name, text_base, attributes, weight=1.2):
        print(f"\n🎭 MULTI-ATTR: {name} ({text_base} + {' + '.join(attributes)})")
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            result_emb = emb_base
            
            for attr in attributes:
                emb_attr = encoder.encode([attr], convert_to_tensor=True, normalize_embeddings=True)
                result_emb = result_emb + (weight * emb_attr)
            
            result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
            
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

    def run_multi_sweep(name, text_base, attributes):
        print(f"\n📊 MULTI-AVG-SWEEP: {name} ({text_base} mixed with {' & '.join(attributes)})")
        weights = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            embs_attr = [encoder.encode([attr], convert_to_tensor=True, normalize_embeddings=True) for attr in attributes]
            
            for w in weights:
                # Moyenne pondérée : (Base + w*Attr1 + w*Attr2) / (1 + num_attr*w)
                result_emb = emb_base
                for emb_attr in embs_attr:
                    result_emb = result_emb + (w * emb_attr)
                
                # Normalisation finale pour rester dans la sphère BGE
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                
                print(f"   [W={w:.1f}] ✨ {sp.decode(tokens[0].tolist())}")

    # --- RUN MULTI SWEEPS ---
    print("\n--- 📊 MULTI-ATTRIBUTE SWEEPS (Phrased Base) ---")
    # On utilise "A car" au lieu de "Car" pour ouvrir l'espace syntaxique
    run_multi_sweep("Le Bolide Rouge", "A car", ["fast", "red"])
    run_multi_sweep("Le Roi-Sorcier", "A man", ["king", "scary"])
    run_multi_sweep("La Tempête Noire", "The sea", ["night", "storm"])

    print("\n--- 🔍 MICRO-SWEEP (Point de Bascule) ---")
    # On zoome sur la zone 0.6 - 0.9
    def run_micro_sweep(name, text_base, attributes):
        print(f"\n🔬 MICRO-SWEEP: {name}")
        weights = [0.65, 0.7, 0.75, 0.8, 0.85]
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            embs_attr = [encoder.encode([attr], convert_to_tensor=True, normalize_embeddings=True) for attr in attributes]
            for w in weights:
                result_emb = emb_base
                for emb_attr in embs_attr:
                    result_emb = result_emb + (w * emb_attr)
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                print(f"   [W={w:.2f}] ✨ {sp.decode(tokens[0].tolist())}")

    run_micro_sweep("Point de Fusion Bolide", "A car", ["fast", "red"])

    def run_impact_sweep(name, text_base, text_impact):
        print(f"\n💥 NARRATIVE IMPACT: {name} ({text_base} + impact '{text_impact}')")
        weights = [0.5, 1.0, 1.5, 2.0]
        with torch.no_grad():
            emb_base = encoder.encode([text_base], convert_to_tensor=True, normalize_embeddings=True)
            emb_impact = encoder.encode([text_impact], convert_to_tensor=True, normalize_embeddings=True)
            
            for w in weights:
                result_emb = emb_base + (w * emb_impact)
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                
                print(f"   [Impact W={w:.1f}] ✨ {sp.decode(tokens[0].tolist())}")

    # --- RUN IMPACT TESTS ---
    print("\n--- 💥 NARRATIVE IMPACTS ---")
    run_impact_sweep("L'Accident", "A red fast car on the highway", "collision")
    run_impact_sweep("L'Incendie", "A beautiful wooden house in the forest", "fire")
    run_impact_sweep("La Fête", "A quiet garden at night", "party")
    run_impact_sweep("Apocalypse", "A futuristic city with tall skyscrapers", "meteor impact")

    def run_narrative_lerp(name, text_before, text_after, steps=10):
        print(f"\n🎬 NARRATIVE LERP: {name}")
        with torch.no_grad():
            emb_start = encoder.encode([text_before], convert_to_tensor=True, normalize_embeddings=True)
            emb_end = encoder.encode([text_after], convert_to_tensor=True, normalize_embeddings=True)
            for i in range(steps):
                alpha = i / (steps - 1)
                result_emb = (1 - alpha) * emb_start + alpha * emb_end
                result_emb = torch.nn.functional.normalize(result_emb, p=2, dim=1)
                
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
                print(f"   [Step {i:02d} α={alpha:.2f}] ✨ {sp.decode(tokens[0].tolist())}")

    print("\n--- 🎬 NARRATIVE TRANSITIONS ---")
    run_narrative_lerp("Le Crash", 
                       "A red fast car on the highway", 
                       "A car crash with broken glass and smoke")
    
    run_narrative_lerp("L'Incendie", 
                       "A beautiful wooden house in the forest", 
                       "A house burning with flames and black smoke")

if __name__ == "__main__":
    surgical_test()
