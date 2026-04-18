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

class SurgicalFineWebDataset(IterableDataset):
    def __init__(self, file_path, seq_len=16):
        self.file_path = file_path
        self.seq_len = seq_len
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count % num_workers == worker_id:
                    text = line.strip()
                    if not text: continue
                    
                    # Tokenisation complète du texte
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    
                    # Découpage en blocs de 16 tokens
                    for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                        segment_ids = tokens[i : i + self.seq_len]
                        # On rajoute l'EOS pour T5
                        segment_ids_with_eos = segment_ids + [self.tokenizer.eos_token_id]
                        
                        # Décodage pour BGE
                        segment_text = self.tokenizer.decode(segment_ids)
                        
                        yield segment_text, torch.tensor(segment_ids_with_eos, dtype=torch.int16)
                count += 1

def build_surgical_chunks():
    parser = argparse.ArgumentParser(description="🎯 Rosetta Surgical 16-Token Forge")
    parser.add_argument('--input-file', type=str, default='../Titan_BGE/data/fineweb_subset.txt')
    parser.add_argument('--output-dir', type=str, default='data/surgical_t5_chunks')
    parser.add_argument('--target-count', type=int, default=1000000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🛰️ Loading BGE Encoder on {device}...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()
    
    dataset = SurgicalFineWebDataset(args.input_file, seq_len=16)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    current_bge = []
    current_labels = []
    total_collected = 0
    chunk_idx = 1
    
    pbar = tqdm(total=args.target_count, desc="🎯 SURGICAL FORGE (16T)")

    for batch_texts, batch_labels in loader:
        # 1. BGE Encoding des fragments de 16 tokens
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
        
        # Sauvegarde
        if len(current_bge) * args.batch_size >= 100000:
            save_path = os.path.join(args.output_dir, f"surgical_part_{chunk_idx}.pt")
            torch.save({
                'bge': torch.cat(current_bge),
                'labels': torch.cat(current_labels)
            }, save_path)
            current_bge, current_labels = [], []
            chunk_idx += 1
            
        if total_collected >= args.target_count:
            break

    pbar.close()
    print(f"✅ Forge terminée ! {total_collected} fragments de 16 tokens prêts.")

if __name__ == "__main__":
    build_surgical_chunks()
