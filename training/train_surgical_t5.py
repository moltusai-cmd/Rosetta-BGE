import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import argparse

from archive.old_scripts.model_t5 import RosettaT5

class FastT5Dataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.all_bge = []
        self.all_labels = []
        
        print(f"📦 Loading {len(self.chunk_files)} surgical chunks...")
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'])
            self.all_labels.append(data['labels'])
            
        self.bge = torch.cat(self.all_bge)
        self.labels = torch.cat(self.all_labels).long()
        print(f"✅ Ready! {len(self.bge)} fragments loaded.")

    def __len__(self): return len(self.bge)
    def __getitem__(self, idx): return self.bge[idx], self.labels[idx]

def train():
    parser = argparse.ArgumentParser(description="⚡ Rosetta-T5 SURGICAL Trainer")
    parser.add_argument('--data-dir', type=str, default='data/surgical_t5_chunks')
    parser.add_argument('--lr', type=float, default=3e-4) # LR un peu plus bas pour la précision
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--compile', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    dataset = FastT5Dataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RosettaT5().to(device)
    if args.compile: model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    scaler = GradScaler(enabled=not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device).float(), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            with autocast(device_type='cuda', dtype=dtype):
                loss, _ = model(bge_embs, target_ids)
                loss = loss / args.grad_accum

            if dtype == torch.float16:
                scaler.scale(loss).backward()
                if (i + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            pbar.set_postfix({"Loss": f"{loss.item() * args.grad_accum:.4f}"})

    torch.save(model.state_dict(), "rosetta_t5_surgical_final.pt")
    print("🏁 Surgical Training Complete!")

if __name__ == "__main__":
    train()
