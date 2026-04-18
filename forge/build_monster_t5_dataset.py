import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

class LocalFineWebDataset(IterableDataset):
    def __init__(self, file_path, target_count=1000000):
        self.file_path = file_path
        self.target_count = target_count
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        count = 0
        valid_count = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Modulo pour paralléliser la lecture/tokenisation
                if count % num_workers == worker_id:
                    text = line.strip()
                    if len(text) > 50:
                        # On tokenize d'abord pour couper proprement à 8 tokens
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(tokens) >= 8:
                            # On prend les 8 premiers tokens pour le BGE et le T5
                            micro_tokens = tokens[:16]
                            text_8 = self.tokenizer.decode(micro_tokens, skip_special_tokens=True)
                            
                            # Labels pour T5 (on garde une petite marge de padding)
                            labels = self.tokenizer.encode(
                                text_8, 
                                max_length=32, 
                                padding='max_length', 
                                truncation=True
                            )
                            yield text_8, torch.tensor(labels, dtype=torch.int16)
                        valid_count += 1
                count += 1
                
                # Note: On ne peut pas facilement s'arrêter ici car on ne connaît pas 
                # la distribution des lignes valides par worker. On gère l'arrêt dans la boucle principale.

def build_t5_chunks_local():
    parser = argparse.ArgumentParser(description="🚀 Rosetta-T5 LOCAL ULTRA Forge")
    parser.add_argument('--input-file', type=str, default='../Titan_BGE/data/fineweb_subset.txt')
    parser.add_argument('--output-dir', type=str, default='data/monster_t5_chunks')
    parser.add_argument('--target-count', type=int, default=1000000)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🛰️ Loading BGE Encoder on {device}...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()
    
    dataset = LocalFineWebDataset(args.input_file, target_count=args.target_count)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    current_bge = []
    current_labels = []
    total_collected = 0
    chunk_idx = 1
    
    pbar = tqdm(total=args.target_count, desc="🔥 RTX 5080 LOCAL FORGE")

    for batch_texts, batch_labels in loader:
        # 1. BGE Encoding (Le GPU va enfin travailler dur)
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
        
        # 2. Sauvegarde par chunks
        if len(current_bge) * args.batch_size >= args.chunk_size:
            save_path = os.path.join(args.output_dir, f"t5_part_{chunk_idx}.pt")
            torch.save({
                'bge': torch.cat(current_bge),
                'labels': torch.cat(current_labels)
            }, save_path)
            
            current_bge = []
            current_labels = []
            chunk_idx += 1
            
        if total_collected >= args.target_count:
            break

    pbar.close()
    print(f"✅ Forge terminée ! {total_collected} exemples prêts dans {args.output_dir}")

if __name__ == "__main__":
    build_t5_chunks_local()
