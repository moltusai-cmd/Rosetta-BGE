import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from archive.old_scripts.model_t5 import RosettaT5
from torch.amp import autocast, GradScaler

class FineWebEduT5Dataset(IterableDataset):
    def __init__(self, target_count=500000):
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        self.target_count = target_count
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def __iter__(self):
        count = 0
        for entry in self.dataset:
            text = entry["text"].strip()
            if len(text) > 30:
                # On tokenize pour couper proprement à 8 tokens
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) >= 8:
                    micro_text = self.tokenizer.decode(tokens[:16], skip_special_tokens=True)
                    yield micro_text
                    count += 1
                    if count >= self.target_count:
                        break

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚜 Monster Larynx Training | FineWeb-Edu | Device: {device}")

    # 1. Composants
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = RosettaT5().to(device)
    
    # On freeze un peu le T5 au début pour protéger ses connaissances ? 
    # Non, pour FineWeb on veut qu'il s'adapte totalement au mapping BGE.
    
    # 2. Data
    target_count = 500000
    dataset = FineWebEduT5Dataset(target_count=target_count)
    loader = DataLoader(dataset, batch_size=48) # Batch size optimisé pour 24GB VRAM, sinon baisser à 32

    # 3. Encodeur BGE
    print("🛰️ Loading BGE Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half()

    # 4. Optimiseur & Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Estimation des steps pour le scheduler
    total_steps = target_count // 48
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    scaler = GradScaler()
    model.train()

    print(f"🌊 Starting the Great Gavage (Target: {target_count} examples)...")
    
    pbar = tqdm(total=total_steps, desc="Larynx Evolution")
    step = 0
    
    for batch_texts in loader:
        # A. BGE Encoding (Inférence mode)
        with torch.no_grad():
            bge_embs = encoder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
            bge_embs = bge_embs.detach().clone().to(device).float()

        # B. T5 Tokenizing
        target_encoding = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=32, 
            return_tensors="pt"
        ).to(device)
        target_ids = target_encoding.input_ids
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        # C. Forward & Backward with Mixed Precision (BF16 if possible)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        with autocast(device_type='cuda', dtype=dtype):
            loss, _ = model(bge_embs, target_ids)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Clip grad norm pour la stabilité
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        pbar.update(1)
        step += 1
        
        # Sauvegarde régulière
        if step % 2000 == 0:
            torch.save(model.state_dict(), f"rosetta_t5_checkpoint_step_{step}.pt")

    print("✅ Monster Larynx Evolution Complete!")
    torch.save(model.state_dict(), "rosetta_t5_fineweb_edu.pt")

if __name__ == "__main__":
    train()
