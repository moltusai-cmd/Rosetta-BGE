import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
from model_v6 import DiffusionRosettaV6
import os

def long_sequence_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
    # Charger Rosetta v6 Ultra
    model = DiffusionRosettaV6(vocab_size=sp.get_piece_size(), d_model=1024, num_cycles=6).to(device)
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        print(f"✅ Brain loaded (Step {ckpt.get('step')})")
    else:
        print(f"❌ Checkpoint {ckpt_path} introuvable.")
        return
        
    model.eval()

    # 1. La phrase longue (64+ tokens)
    long_text = ("The ancient city of Rome was not built in a single day, but through centuries of "
                 "architectural innovation, political struggle, and the relentless ambition of its "
                 "citizens who sought to create an eternal empire that would eventually span across "
                 "three continents and define the course of Western civilization for millennia to come.")
    
    full_tokens = sp.encode(long_text)
    print(f"\n📝 Long Sequence Length: {len(full_tokens)} tokens")
    
    # 2. Paramètres de la fenêtre
    window_size = 16
    stride = 8
    
    # Préparation du buffer de sortie (on va superposer et voter)
    reconstruction = [[] for _ in range(len(full_tokens))]
    
    print(f"🔄 Sliding through windows (Size={window_size}, Stride={stride})...")
    
    for start in range(0, len(full_tokens) - window_size + 1, stride):
        end = start + window_size
        chunk_tokens = full_tokens[start:end]
        chunk_text = sp.decode(chunk_tokens)
        
        # Encodage BGE du fragment
        with torch.no_grad():
            emb = encoder.encode([chunk_text], convert_to_tensor=True, normalize_embeddings=True)
            # Décodage Rosetta
            decoded_ids = model.decode(emb, num_steps=24)[0].tolist()
            
        # Affichage pour voir ce que chaque fenêtre "comprend"
        print(f"   Window [{start:02d}:{end:02d}] ➔ ✨ {sp.decode(decoded_ids)}")
            
        # Placement dans la reconstruction
        for i, tid in enumerate(decoded_ids):
            if start + i < len(full_tokens) and tid != sp.eos_id(): # On ignore les EOS générés s'ils y en a
                reconstruction[start + i].append(tid)

    # 3. Assemblage Final (Vote majoritaire simple sur les chevauchements)
    final_ids = []
    for slots in reconstruction:
        if not slots: continue
        # On prend le token le plus fréquent prédit pour cette position
        most_common = max(set(slots), key=slots.count)
        final_ids.append(most_common)
        
    reconstructed_text = sp.decode(final_ids)
    
    print("\n--- 🏁 RESULTATS DU TISSAGE ---")
    print(f"🎯 Target : {long_text}")
    print(f"✨ Rosetta: {reconstructed_text}")
    
    # Calcul de la perfection (Match exact par position)
    # On compare sur la longueur de ce qui a pu être couvert par les fenêtres
    comparable_length = len([s for s in reconstruction if s])
    matches = sum(1 for a, b in zip(full_tokens[:comparable_length], final_ids) if a == b)
    if comparable_length > 0:
        accuracy = (matches / comparable_length) * 100
        print(f"\n📊 Tissage Accuracy (Exact Match): {accuracy:.2f}%")

    # --- NOUVEAU TEST : LE RÉSUMÉ GLOBAL ---
    print("\n🌀 --- GLOBAL SUMMARY TEST (66 to 16 tokens) ---")
    with torch.no_grad():
        # On encode la phrase COMPLETE (66 tokens) dans UN SEUL vecteur
        global_emb = encoder.encode([long_text], convert_to_tensor=True, normalize_embeddings=True)
        
        # On demande à Rosetta de décompresser ce vecteur saturé
        summary_ids = model.decode(global_emb, num_steps=24)[0].tolist()
        summary_text = sp.decode(summary_ids)
        
        print(f"🎯 Full Target  (66 tokens): {long_text}")
        print(f"✨ Rosetta Summary (16 tokens): {summary_text}")

    # --- TEST DE COMPRESSION EXTRÊME (128 TOKENS) ---
    print("\n🌋 --- ULTRA-COMPRESSION TEST (128 to 16 tokens) ---")
    ultra_long_text = ("The Industrial Revolution, which took place from the 18th to 19th centuries, "
                       "was a period during which predominantly agrarian, rural societies in Europe "
                       "and America became industrial and urban. Prior to the Industrial Revolution, "
                       "which began in Britain in the late 1700s, manufacturing was often done in "
                       "people’s homes, using hand tools or basic machines. Industrialization marked "
                       "a shift to powered, special-purpose machinery, factories and mass production. "
                       "The iron and textile industries, along with the development of the steam engine, "
                       "played central roles in the Industrial Revolution, which also saw improved "
                       "systems of transportation, communication and banking. While industrialization "
                       "brought about an increased volume and variety of manufactured goods and an "
                       "improved standard of living for some, it also resulted in often grim "
                       "employment and living conditions for the poor and working classes.")
    
    ultra_tokens = sp.encode(ultra_long_text)
    print(f"📝 Ultra-Long Sequence Length: {len(ultra_tokens)} tokens")

    with torch.no_grad():
        # Encodage du monstre (128 tokens)
        ultra_emb = encoder.encode([ultra_long_text], convert_to_tensor=True, normalize_embeddings=True)
        
        # Décompression Rosetta
        ultra_summary_ids = model.decode(ultra_emb, num_steps=24)[0].tolist()
        ultra_summary_text = sp.decode(ultra_summary_ids)
        
        print(f"\n✨ Rosetta Ultra-Summary (16 tokens): {ultra_summary_text}")

    # --- TEST DU TROU NOIR (1024 TOKENS) ---
    print("\n🕳️ --- BLACK HOLE TEST (1024 to 16 tokens) ---")
    # On crée un texte massif en répétant et variant les thèmes de l'IA
    ai_history_base = ("Artificial Intelligence (AI) began with Alan Turing’s question: Can machines think? "
                       "From the 1950s Dartmouth Workshop to the 'AI Winters' of the 70s and 80s, the field "
                       "struggled with limited computing power. The 1990s saw IBM’s Deep Blue defeat Gary Kasparov. "
                       "The 2010s marked the rise of Deep Learning and Neural Networks, fueled by big data. "
                       "Today, Large Language Models and Generative AI are transforming every industry. "
                       "We are now moving toward Artificial General Intelligence (AGI) and beyond. ")
    
    # On multiplie pour atteindre les 1024 tokens environ
    black_hole_text = ai_history_base * 12 
    
    bh_tokens = sp.encode(black_hole_text)
    print(f"📝 Black Hole Sequence Length: {len(bh_tokens)} tokens")

    with torch.no_grad():
        # Encodage du Trou Noir (Saturé à 1024 tokens)
        bh_emb = encoder.encode([black_hole_text], convert_to_tensor=True, normalize_embeddings=True)
        
        # Décompression Rosetta
        bh_summary_ids = model.decode(bh_emb, num_steps=24)[0].tolist()
        bh_summary_text = sp.decode(bh_summary_ids)
        
        print(f"\n✨ Rosetta Black Hole Summary (16 tokens): {bh_summary_text}")

    # --- NOUVELLE BATTERIE : SATURATION CONCEPTUELLE (512 TOKENS) ---
    def run_saturation_test(name, full_text):
        print(f"\n🧪 --- SATURATION TEST: {name} ---")
        tokens = sp.encode(full_text)
        print(f"📝 Input Length: {len(tokens)} tokens (Target: ~512)")
        
        with torch.no_grad():
            emb = encoder.encode([full_text], convert_to_tensor=True, normalize_embeddings=True)
            decoded_ids = model.decode(emb, num_steps=24)[0].tolist()
            summary = sp.decode(decoded_ids)
            print(f"✨ Rosetta Master Signal (16 tokens): {summary}")

    # 1. PHYSIQUE QUANTIQUE
    quantum_text = ("Quantum mechanics is a fundamental theory in physics that provides a description of the "
                    "physical properties of nature at the scale of atoms and subatomic particles. It is the "
                    "foundation of all quantum physics including quantum chemistry, quantum field theory, "
                    "quantum technology, and quantum information science. Classical physics, the collection "
                    "of theories that existed before the advent of quantum mechanics, describes many aspects "
                    "of nature at an ordinary scale, but is not sufficient for describing them at very small "
                    "scales. Most theories in classical physics can be derived from quantum mechanics as an "
                    "approximation valid at large scale. Quantum mechanics differs from classical physics in "
                    "that energy, momentum, angular momentum, and other quantities of a bound system are "
                    "restricted to discrete values (quantization), objects have characteristics of both "
                    "particles and waves (wave-particle duality), and there are limits to how accurately "
                    "the value of a physical quantity can be predicted prior to its measurement, given a "
                    "complete set of initial conditions (the uncertainty principle).") * 4 # Pour atteindre ~500 tokens
    
    run_saturation_test("QUANTUM PHYSICS", quantum_text)

    # 2. ÉVOLUTION BIOLOGIQUE
    bio_text = ("Evolution is change in the heritable characteristics of biological populations over "
                "successive generations. These characteristics are the expressions of genes that are "
                "passed on from parent to offspring during reproduction. Different characteristics "
                "tend to exist within any given population as a result of mutation, genetic recombination "
                "and other sources of genetic variation. Evolution occurs when evolutionary processes "
                "such as natural selection (including sexual selection) and genetic drift act on this "
                "variation, resulting in certain characteristics becoming more common or rare within a "
                "population. It is this process of evolution that has given rise to biodiversity at "
                "every level of biological organisation, including the levels of species, individual "
                "organisms and molecules. All life on Earth shares a last universal common ancestor "
                "that lived approximately 3.5–3.8 billion years ago.") * 4
    
    run_saturation_test("BIOLOGICAL EVOLUTION", bio_text)

    # 3. EXPLORATION SPATIALE
    space_text = ("The Voyager 1 spacecraft is a 722-kilogram space probe launched by NASA on September 5, 1977, "
                  "to study the outer Solar System. Having operated for 46 years, 6 months and 18 days as of "
                  "March 23, 2024, it still communicates with the Deep Space Network to receive routine "
                  "commands and to transmit data to Earth. At a distance of 162.7 AU from Earth as of "
                  "February 2024, it is the most distant human-made object from Earth. The probe's "
                  "objectives included flybys of Jupiter, Saturn and Saturn's largest moon, Titan. "
                  "While its brother probe Voyager 2 also visited Uranus and Neptune, Voyager 1 was the "
                  "first of the two probes to leave the heliosphere and enter the interstellar medium. "
                  "It carries a gold-plated audio-visual disc, the Golden Record, containing sounds and "
                  "images selected to portray the diversity of life and culture on Earth.") * 4
    
    run_saturation_test("SPACE EXPLORATION", space_text)

    # 4. PHILOSOPHIE DE LA CONSCIENCE
    phil_text = ("The hard problem of consciousness is the problem of explaining why and how we have "
                 "qualitative phenomenal experiences. It is contrasted with the 'easy problems' of "
                 "explaining the physical mechanisms that give rise to biological function and behavior. "
                 "David Chalmers, who introduced the term 'hard problem', argues that even if we explain "
                 "all the functional and structural facts about the brain, there remains a further "
                 "question: Why is the performance of these functions accompanied by experience? "
                 "This gap between physical processes and subjective experience is known as the "
                 "explanatory gap. Materialist theories attempt to reduce consciousness to brain states, "
                 "while dualist theories suggest that mind and matter are fundamentally different. "
                 "Panpsychism offers a third view, suggesting that consciousness is a fundamental "
                 "property of all matter in the universe.") * 4
    
    run_saturation_test("PHILOSOPHY OF MIND", phil_text)

if __name__ == "__main__":
    long_sequence_test()
