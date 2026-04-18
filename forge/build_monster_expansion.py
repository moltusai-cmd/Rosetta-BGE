import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import T5TokenizerFast
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Configuration du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MonsterStreamDataset(IterableDataset):
    def __init__(self, seq_len=16):
        self.seq_len = seq_len
        # On définit une longueur maximale immense pour éviter les warnings, 
        # car on gère le découpage en 16 tokens manuellement après.
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small", model_max_length=1000000)

    def __iter__(self):
        # On charge le stream complet (10BT)
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        count = 0
        for entry in ds:
            # Répartition entre les cœurs CPU
            if count % num_workers == worker_id:
                text = entry["text"].strip()
                if not text: continue
                
                # Tokenization rapide via Rust
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                    segment_ids = tokens[i : i + self.seq_len]
                    segment_text = self.tokenizer.decode(segment_ids)
                    yield segment_text, torch.tensor(segment_ids + [self.tokenizer.eos_token_id], dtype=torch.int16)
            count += 1

def build_monster_expansion():
    parser = argparse.ArgumentParser(description="🌋 Rosetta Monster Dataset Expansion")
    parser.add_argument('--target-count', type=int, default=3000000, help="Nombre de fragments (3M ≈ 15GB texte)")
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    output_dir = 'data/surgical_monster_chunks'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"🛰️  Loading BGE Master Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()
    
    dataset = MonsterStreamDataset(seq_len=16)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    current_bge, current_labels = [], []
    total_collected = 0
    chunk_idx = 1
    
    # On commence après les chunks existants
    existing = glob.glob("data/surgical_t5_chunks/*.pt")
    chunk_offset = len(existing) + 1

    pbar = tqdm(total=args.target_count, desc="🌊 STREAMING MONSTER DATA")

    try:
        for batch_texts, batch_labels in loader:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    embs = encoder.encode(
                        list(batch_texts), 
                        convert_to_tensor=True, 
                        normalize_embeddings=True,
                        batch_size=args.batch_size, 
                        show_progress_bar=False
                    )
                    current_bge.append(embs.half().cpu())
            
            current_labels.append(batch_labels)
            total_collected += len(batch_texts)
            pbar.update(len(batch_texts))
            
            # On sauvegarde par blocs de 100k
            if total_collected % 100000 < args.batch_size:
                save_path = os.path.join(output_dir, f"monster_part_{chunk_idx + chunk_offset}.pt")
                torch.save({
                    'bge': torch.cat(current_bge),
                    'labels': torch.cat(current_labels)
                }, save_path)
                current_bge, current_labels = [], []
                chunk_idx += 1
                
            if total_collected >= args.target_count:
                break
    except KeyboardInterrupt:
        print("\n🛑 Extraction interrupted.")

    pbar.close()
    print(f"✅ Gavage terminé ! {total_collected} nouveaux fragments ajoutés.")

if __name__ == "__main__":
    import glob
    build_monster_expansion()
