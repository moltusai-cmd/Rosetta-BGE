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
        
        print(f"📦 Loading {len(self.chunk_files)} chunks into VRAM/RAM...")
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'])
            self.all_labels.append(data['labels'])
            
        self.bge = torch.cat(self.all_bge)
        self.labels = torch.cat(self.all_labels).long()
        print(f"✅ Ready! {len(self.bge)} examples loaded.")

    def __len__(self):
        return len(self.bge)

    def __getitem__(self, idx):
        return self.bge[idx], self.labels[idx]

def train():
    parser = argparse.ArgumentParser(description="⚡ Rosetta-T5 FLASH Trainer")
    parser.add_argument('--data-dir', type=str, default='data/monster_t5_chunks')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--compile', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Vérification BF16 pour la RTX 5080
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"🚀 Device: {device} | BF16: {use_bf16} | Compile: {args.compile}")

    # 1. Load Pre-computed Data
    dataset = FastT5Dataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 2. Setup Rosetta-T5
    print("🏗️ Forge du Larynx Rosetta-T5...")
    model = RosettaT5().to(device)
    
    if args.compile:
        print("⚡ Compiling T5 kernels with torch.compile...")
        try:
            # Mode reduce-overhead consomme beaucoup de VRAM, on peut passer en default si besoin
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ Compile failed, skipping: {e}")

    # 3. Optim & Schedule
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    scaler = GradScaler(enabled=not use_bf16) 
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 4. Training Loop
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device).float(), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            with autocast(device_type='cuda', dtype=dtype):
                loss, _ = model(bge_embs, target_ids)
                loss = loss / args.grad_accum # Normalisation pour l'accumulation

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

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Checkpoint par époque
        torch.save(model.state_dict(), f"rosetta_t5_larynx_epoch_{epoch+1}.pt")

    print("🏁 Training Finished! Model saved as rosetta_t5_larynx_final.pt")
    torch.save(model.state_dict(), "rosetta_t5_larynx_final.pt")

if __name__ == "__main__":
    train()
