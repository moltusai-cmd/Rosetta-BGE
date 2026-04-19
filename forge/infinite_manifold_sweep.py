import os
import sys
import torch
import torch.nn.functional as F
from transformers import T5TokenizerFast, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
import glob
import warnings
import queue
import threading

# Configuration du chemin racine pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v6_pro import RosettaV6Pro

# Désactiver les avertissements
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def master_manifold_sweep_v4():
    device = torch.device("cuda")
    
    # --- CONFIGURATION MASTER FORGE (V4) ---
    TARGET_SIMILARITY = 0.96 # On monte la barre car le modèle est meilleur !
    BATCH_SIZE = 128        
    POPULATION_SIZE = 32   
    SAVE_EVERY = 5000
    
    print(f"🌋 MASTER FORGE V4.0 | ROSETTA V6 PRO 🌋")
    print(f"Target Similarity: > {TARGET_SIMILARITY}")

    t5_path = 't5-small'
    bge_path = 'BAAI/bge-small-en-v1.5'
    
    # 1. Modèle Rosetta V6 PRO Master
    model = RosettaV6Pro(num_guides=16).to(device)
    checkpoint = 'checkpoints/rosetta_v6_pro_master.pt'
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})
    model.half().eval() # Tout Rosetta en FP16 pour la vitesse

    print("⚖️ Loading & Compiling BGE Judge...")
    bge_model = AutoModel.from_pretrained(bge_path).to(device).half()
    bge_model = torch.compile(bge_model)
    bge_model.eval()

    tokenizer = T5TokenizerFast.from_pretrained(t5_path)
    bge_tokenizer = AutoTokenizer.from_pretrained(bge_path)

    # 2. Données Ancre (4 Millions de points possibles)
    chunk_files = sorted(glob.glob("data/surgical_monster_chunks/*.pt")) + sorted(glob.glob("data/surgical_t5_chunks/*.pt"))
    print(f"📦 Loading {len(chunk_files)} anchor chunks...")
    all_vectors = [torch.load(f, map_location='cpu', weights_only=True)['bge'] for f in chunk_files[:30]]
    fineweb_bge = torch.cat(all_vectors).to(device).half()
    
    os.makedirs("data/manifold_gold_v4", exist_ok=True)
    file_index = len(glob.glob("data/manifold_gold_v4/*.pt")) + 1
    
    gen_queue = queue.Queue(maxsize=2)
    
    def generator_thread_func():
        while True:
            with torch.no_grad():
                idx_a = torch.randint(0, len(fineweb_bge), (BATCH_SIZE,))
                idx_b = torch.randint(0, len(fineweb_bge), (BATCH_SIZE,))
                alpha = torch.rand(BATCH_SIZE, 1, device=device).half()
                target_vec = F.normalize((1 - alpha) * fineweb_bge[idx_a] + alpha * fineweb_bge[idx_b], p=2, dim=1)
                
                # Rosetta PRO Generation
                # Tout est déjà en half (modèle et target_vec)
                guides = model.get_guides(target_vec)
                
                output_ids = model.t5.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=guides),
                    max_length=20, do_sample=True, top_k=50, num_return_sequences=POPULATION_SIZE
                )
                gen_queue.put((target_vec.cpu(), output_ids.cpu()))

    threading.Thread(target=generator_thread_func, daemon=True).start()

    gold_bge_list, gold_labels_list, total_gold_mined = [], [], 0
    
    print("🚀 Master Forge Started. Harvesting Ultra-Pépites...")
    
    try:
        while True: 
            target_vec_cpu, output_ids_cpu = gen_queue.get()
            
            with torch.no_grad():
                phrases = tokenizer.batch_decode(output_ids_cpu, skip_special_tokens=True)
                encoded = bge_tokenizer(phrases, padding=True, truncation=True, return_tensors='pt', max_length=32).to(device)
                
                all_embs = []
                for j in range(0, len(encoded['input_ids']), 2048):
                    sub_ids = encoded['input_ids'][j : j + 2048]
                    sub_mask = encoded['attention_mask'][j : j + 2048]
                    out = bge_model(input_ids=sub_ids, attention_mask=sub_mask)
                    all_embs.append(F.normalize(out[0][:, 0], p=2, dim=1).float())
                
                phrases_embs = torch.cat(all_embs).view(BATCH_SIZE, POPULATION_SIZE, 384)
                target_vec_exp = target_vec_cpu.float().to(device).unsqueeze(1).expand(-1, POPULATION_SIZE, -1)
                similarities = F.cosine_similarity(target_vec_exp, phrases_embs, dim=2)
                
                best_sims, best_indices = torch.max(similarities, dim=1)
                gold_mask = best_sims > TARGET_SIMILARITY
                
                if gold_mask.sum() > 0:
                    found = gold_mask.sum().item()
                    total_gold_mined += found
                    
                    gold_mask_cpu = gold_mask.cpu()
                    gold_bge_list.append(target_vec_cpu[gold_mask_cpu])
                    
                    g_idx = torch.arange(BATCH_SIZE)[gold_mask_cpu] * POPULATION_SIZE + best_indices[gold_mask].cpu()
                    g_ids = output_ids_cpu[g_idx]
                    
                    padded = torch.zeros((g_ids.size(0), 25), dtype=torch.long)
                    padded[:, :min(g_ids.size(1), 25)] = g_ids[:, :min(g_ids.size(1), 25)]
                    gold_labels_list.append(padded)
                    
                    if total_gold_mined % 100 < found:
                        print(f"💎 Master Gold: {total_gold_mined} | Sim: {best_sims[gold_mask].max().item():.4f} | '{phrases[g_idx[0]]}'")

                if sum(t.size(0) for t in gold_bge_list) >= SAVE_EVERY:
                    out_path = f"data/manifold_gold_v4/master_gold_{file_index}.pt"
                    torch.save({"bge": torch.cat(gold_bge_list), "labels": torch.cat(gold_labels_list)}, out_path)
                    print(f"💾 Saved gold chunk {file_index} ({total_gold_mined} total)")
                    gold_bge_list, gold_labels_list, file_index = [], [], file_index + 1

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    master_manifold_sweep_v4()
