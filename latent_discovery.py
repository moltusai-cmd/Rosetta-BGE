import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import random

def genetic_discovery():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    # 1. Définir le Vecteur Cible (L'étape 0.75 de notre interpolation)
    text_a = "A wise old man sitting in a quiet library reading a book"
    text_b = "A massive black hole consuming a bright star in deep space"
    
    with torch.no_grad():
        vec_a = encoder.encode([text_a], convert_to_tensor=True, normalize_embeddings=True).float()
        vec_b = encoder.encode([text_b], convert_to_tensor=True, normalize_embeddings=True).float()
        # Le point critique à 0.75
        target_vec = F.normalize(0.25 * vec_a + 0.75 * vec_b, p=2, dim=1)

    # 2. Algorithme Génétique sur le Texte
    # Dictionnaire de briques sémantiques pour la mutation
    words = ["massive", "deep", "black", "hole", "star", "consuming", "devouring", "space", "dark", "bright", "star", "swallowing", "infinity", "void", "cosmos", "light", "shadow", "ancient", "cosmic"]
    
    # Population initiale (basée sur ce que Rosetta a bafouillé)
    population = [
        "a deep star in a dark hole consuming a billions",
        "a black hole devouring a bright star in space",
        "massive black hole swallowing light in the void",
        "a dark star falling into a massive hole in deep space",
        "cosmic event where a star is consumed by a hole"
    ]

    print(f"🎯 Target Acquired. Searching for the 'Perfect Phrase'...")

    best_sim = 0
    best_phrase = ""

    for generation in range(50):
        scored_pop = []
        for phrase in population:
            with torch.no_grad():
                emb = encoder.encode([phrase], convert_to_tensor=True, normalize_embeddings=True).float()
                sim = F.cosine_similarity(emb, target_vec).item()
                scored_pop.append((sim, phrase))
                
                if sim > best_sim:
                    best_sim = sim
                    best_phrase = phrase

        # Tri et sélection des 5 meilleurs
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        top_phrases = [p[1] for p in scored_pop[:5]]

        if best_sim > 0.99: # On a trouvé le Graal
            break

        # Mutation et Croisement
        new_population = list(top_phrases)
        while len(new_population) < 20:
            parent = random.choice(top_phrases)
            parts = parent.split()
            # Mutation : on change, ajoute ou supprime un mot
            if random.random() > 0.5 and len(parts) > 2:
                idx = random.randint(0, len(parts)-1)
                parts[idx] = random.choice(words)
            else:
                parts.append(random.choice(words))
            
            new_phrase = " ".join(parts)
            new_population.append(new_phrase)
        
        population = new_population
        if generation % 10 == 0:
            print(f"🧬 Gen {generation} | Best Sim: {best_sim:.4f} | '{best_phrase}'")

    print(f"\n🏆 Discovery Complete!")
    print(f"✨ Best Phrase : '{best_phrase}'")
    print(f"📊 Similarity  : {best_sim:.4f}")
    
    # On sauvegarde cette "vérité" pour Rosetta
    discovery_data = {
        "target_vec": target_vec.cpu(),
        "perfect_phrase": best_phrase,
        "similarity": best_sim
    }
    torch.save(discovery_data, "latent_discovery_gold.pt")

if __name__ == "__main__":
    genetic_discovery()
