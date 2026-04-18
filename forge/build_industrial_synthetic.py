import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import glob

from core.model_v6 import RosettaV6

def monster_forge():
    device = torch.device("cuda")
    # CONFIGURATION MONSTRUEUSE pour RTX 5080
    BATCH_SIZE = 2048 # Saturation CUDA
    TOTAL_GOLD_TARGET = 1000000 # 1 MILLION de concepts parfaits
    THRESHOLD = 0.95 
    CHUNK_SIZE = 50000 

    print(f"🌋 MONSTER FORGE ACTIVATED | Target: 1M Gold | Batch: {BATCH_SIZE}")

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = RosettaV6(num_guides=16).to(device)
    state_dict = torch.load('rosetta_v6_epoch_25.pt', map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # Chargement BGE Raw pour performance maximale
    print("🧠 Loading BGE Model for judging...")
    bge_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device).half()
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    bge_model.eval()

    # Chargement massif de la matière première
    chunk_files = sorted(glob.glob("data/surgical_t5_chunks/*.pt"))
    print(f"📦 Loading base vectors for infinite mixing...")
    all_vectors = []
    for f in chunk_files[:20]: 
        d = torch.load(f, map_location='cpu', weights_only=True)
        all_vectors.append(d['bge'])
    real_bge = torch.cat(all_vectors).float().to(device)
    
    os.makedirs("data/synthetic_gold", exist_ok=True)

    gold_bge_list = []
    gold_labels_list = []
    total_found = 0
    chunk_count = 1
    current_gold_count = 0
    
    pbar = tqdm(total=TOTAL_GOLD_TARGET, desc="💎 Forging 1M Gold Concepts")
    
    while total_found < TOTAL_GOLD_TARGET:
        with torch.no_grad():
            # 1. Mixage Latent
            idx_a = torch.randint(0, len(real_bge), (BATCH_SIZE,))
            idx_b = torch.randint(0, len(real_bge), (BATCH_SIZE,))
            alpha = torch.rand(BATCH_SIZE, 1).to(device)
            target_vec = F.normalize((1 - alpha) * real_bge[idx_a] + alpha * real_bge[idx_b], p=2, dim=1)
            
            # 2. Rosetta Turbo Generation (Greedy)
            x = model.input_norm(target_vec)
            guides = model.projector(x).view(BATCH_SIZE, 16, model.d_model)
            
            output_ids = model.t5.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=guides),
                max_length=20,
                num_beams=1, 
                do_sample=False
            )
            
            # 3. Validation Sémantique Ultra-Rapide
            phrases = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Tokenization BGE batchée
            encoded_input = bge_tokenizer(phrases, padding=True, truncation=True, return_tensors='pt', max_length=32).to(device)
            model_output = bge_model(**encoded_input)
            reconstructed_embs = F.normalize(model_output[0][:, 0], p=2, dim=1).float()
            
            similarities = F.cosine_similarity(target_vec, reconstructed_embs)
            gold_mask = similarities > THRESHOLD
            
            if gold_mask.sum() > 0:
                num_found = gold_mask.sum().item()
                gold_bge_list.append(target_vec[gold_mask].cpu())
                
                g_ids = output_ids[gold_mask]
                padded_ids = torch.full((g_ids.size(0), 25), 0, dtype=torch.long)
                s_len = min(g_ids.size(1), 25)
                padded_ids[:, :s_len] = g_ids[:, :s_len].cpu()
                gold_labels_list.append(padded_ids)
                
                total_found += num_found
                current_gold_count += num_found
                pbar.update(num_found)

        # Sauvegarde par morceaux (Checkpointing)
        if current_gold_count >= CHUNK_SIZE:
            out_path = f"data/synthetic_gold/monster_gold_{chunk_count}.pt"
            torch.save({
                "bge": torch.cat(gold_bge_list),
                "labels": torch.cat(gold_labels_list)
            }, out_path)
            gold_bge_list = []
            gold_labels_list = []
            current_gold_count = 0
            chunk_count += 1

    pbar.close()
    print(f"🏁 DONE! 1 Million Gold Concepts Forged.")

if __name__ == "__main__":
    monster_forge()
