import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
from archive.old_scripts.model_t5 import RosettaT5

class T5RosettaDataset(Dataset):
    def __init__(self, tokenizer, target_tokens=100000):
        from datasets import load_dataset
        print("📥 Loading wikitext-103 (Streaming)...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        
        self.tokenizer = tokenizer
        self.data = []
        
        count = 0
        for entry in dataset:
            text = entry["text"].strip()
            if len(text) < 50: continue
            
            # On prend des segments de texte courts pour Rosetta-T5 (Larynx)
            self.data.append(text)
            count += 1
            if count >= target_tokens // 20: break # Approximatif

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Rosetta-T5 Larynx Training Suite | Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5().to(device)
    
    # Dataset & Loader
    dataset = T5RosettaDataset(tokenizer, target_tokens=100000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Encodeur BGE (pour la condition)
    print("🛰️ Loading BGE Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(3):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/3")
        for batch_texts in pbar:
            # 1. BGE Encoding
            with torch.no_grad():
                bge_embs = encoder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
                # Clone indispensable pour sortir du mode inférence et passage en float pour le projecteur
                bge_embs = bge_embs.detach().clone().to(device).float()

            # 2. T5 Tokenizing
            target_encoding = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=32, 
                return_tensors="pt"
            ).to(device)
            target_ids = target_encoding.input_ids
            target_ids[target_ids == tokenizer.pad_token_id] = -100 # Standard T5 masking

            # 3. Forward
            loss, logits = model(bge_embs, target_ids)

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    print("✅ Rosetta-T5 Larynx Trained!")
    torch.save(model.state_dict(), "rosetta_t5_larynx.pt")

if __name__ == "__main__":
    train()
