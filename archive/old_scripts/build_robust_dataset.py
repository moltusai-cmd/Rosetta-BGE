import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse

class SimpleBPETokenizer:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.inv_vocab = {}
        self.mask_token_id = vocab_size 
        
    def encode(self, text):
        tokens = []
        for word in text.split():
            h = 0
            for char in word: h = (h * 31 + ord(char)) % self.vocab_size
            tokens.append(h)
            if h not in self.inv_vocab: self.inv_vocab[h] = word
        return tokens
        
    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(int(tid), f"") for tid in token_ids if int(tid) < self.vocab_size])

def build_robust_chunks():
    parser = argparse.ArgumentParser(description="🚜 Rosetta Robust Dataset Pre-calculator")
    parser.add_argument('--output-dir', type=str, default='data/robust_chunks')
    parser.add_argument('--target-tokens', type=int, default=1000000, help="Total tokens to harvest")
    parser.add_argument('--chunk-size', type=int, default=50000, help="Tokens per chunk file")
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("🛰️ Loading BGE Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()
    
    tokenizer = SimpleBPETokenizer(16384)
    
    from datasets import load_dataset
    print("📥 Loading wikitext-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    current_tokens = []
    chunk_idx = 1
    total_tokens_collected = 0
    seq_len = 16

    pbar = tqdm(total=args.target_tokens, desc="🌾 Harvesting & Encoding")

    def save_chunk(tokens_list, idx):
        # We need to compute BGE for each 16-token segment
        num_segments = len(tokens_list) // seq_len
        segments = []
        for i in range(num_segments):
            segments.append(tokens_list[i * seq_len : (i + 1) * seq_len])
        
        segments_tensor = torch.tensor(segments, dtype=torch.long)
        
        # Decode and encode in batches
        all_embeddings = []
        for i in range(0, len(segments), args.batch_size):
            batch = segments[i : i + args.batch_size]
            batch_texts = [tokenizer.decode(s) for s in batch]
            with torch.no_grad():
                embs = encoder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
                all_embeddings.append(embs.half().cpu())
        
        save_path = os.path.join(args.output_dir, f"robust_part_{idx}.pt")
        torch.save({
            'embeddings': torch.cat(all_embeddings),
            'token_ids': segments_tensor
        }, save_path)
        print(f"✅ Saved chunk {idx} to {save_path}")

    for entry in dataset:
        text = entry["text"].strip()
        if not text: continue
        
        toks = tokenizer.encode(text)
        current_tokens.extend(toks)
        
        while len(current_tokens) >= args.chunk_size:
            chunk_to_save = current_tokens[:args.chunk_size]
            save_chunk(chunk_to_save, chunk_idx)
            
            total_tokens_collected += len(chunk_to_save)
            pbar.update(len(chunk_to_save))
            
            current_tokens = current_tokens[args.chunk_size:]
            chunk_idx += 1
            
            if total_tokens_collected >= args.target_tokens:
                break
        
        if total_tokens_collected >= args.target_tokens:
            break

    pbar.close()
    print(f"🏁 Done! Collected {total_tokens_collected} tokens across {chunk_idx-1} chunks.")

if __name__ == "__main__":
    build_robust_chunks()
