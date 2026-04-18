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
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'])
            self.all_labels.append(data['labels'])
        self.bge = torch.cat(self.all_bge)
        self.labels = torch.cat(self.all_labels).long()

    def __len__(self): return len(self.bge)
    def __getitem__(self, idx): return self.bge[idx], self.labels[idx]

def train():
    parser = argparse.ArgumentParser(description="⚡ Rosetta-T5 PRO Surgical Trainer")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128) # Batch plus petit car plus de mémoire (Mirror head)
    parser.add_argument('--grad-accum', type=int, default=4)   # Batch effectif 512
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    dataset = FastT5Dataset('data/surgical_t5_chunks')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RosettaT5().to(device)
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    scaler = GradScaler(enabled=not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"🔥 Training Rosetta-T5 PRO | Epochs: {args.epochs} | Device: {device}")

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device).float(), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            with autocast(device_type='cuda', dtype=dtype):
                loss_ce, _, bge_recon = model(bge_embs, target_ids)
                
                # Loss Miroir : MSE entre le BGE original et reconstruit
                loss_mirror = F.mse_loss(bge_recon, bge_embs)
                
                # Perte Totale : On donne beaucoup de poids au miroir pour la fidélité
                loss = (loss_ce + 1.0 * loss_mirror) / args.grad_accum

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

            pbar.set_postfix({"L_CE": f"{loss_ce.item():.4f}", "L_MIR": f"{loss_mirror.item():.4f}"})

        # Sauvegarde progressive
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"rosetta_t5_pro_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "rosetta_t5_pro_final.pt")
    print("🏁 PRO Surgical Training Complete!")

if __name__ == "__main__":
    train()
