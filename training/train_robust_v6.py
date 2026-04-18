import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import argparse

from core.model_v6 import RosettaV6

class RobustDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.all_bge = []
        self.all_labels = []
        print("📦 Loading Fragments for V6...")
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'])
            self.all_labels.append(data['labels'])
        self.bge = torch.cat(self.all_bge).float()
        self.labels = torch.cat(self.all_labels).long()
        print(f"✅ Ready! {len(self.bge)} fragments loaded.")

    def __len__(self): return len(self.bge)
    def __getitem__(self, idx): return self.bge[idx], self.labels[idx]

def train():
    parser = argparse.ArgumentParser(description="💎 Rosetta-V6 Robust Trainer")
    parser.add_argument('--epochs', type=int, default=40) # On allonge pour la perfection
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--grad-accum', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RosettaV6().to(device)
    model = torch.compile(model)

    dataset = RobustDataset('data/surgical_t5_chunks')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01) # LR plus bas pour la précision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = GradScaler(enabled=not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"🔥 Training Rosetta-V6 | High-Res Mode | BF16: {use_bf16}")

    # Mirror targets (pour la perte cosinus)
    # CosineEmbeddingLoss attend un label '1' pour signifier 'être identique'
    cosine_loss_fn = nn.CosineEmbeddingLoss()

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            # DATA AUGMENTATION : Léger bruit gaussien pour la robustesse (Titan drift defense)
            if model.training:
                noise = torch.randn_like(bge_embs) * 0.005
                bge_embs_noisy = F.normalize(bge_embs + noise, p=2, dim=1)
            else:
                bge_embs_noisy = bge_embs

            with autocast(device_type='cuda', dtype=dtype):
                loss_ce, _, bge_recon = model(bge_embs_noisy, target_ids)
                
                # Perte de similarité Cosinus (Angle parfait)
                # target=1 dit qu'on veut que les vecteurs soient alignés
                target = torch.ones(bge_embs.size(0)).to(device)
                loss_cosine = cosine_loss_fn(bge_recon, bge_embs, target)
                
                # Perte combinée (Fidélité linguistique + Ancrage sémantique)
                loss = (loss_ce + 5.0 * loss_cosine) / args.grad_accum # On booste le poids du cosinus

            if dtype == torch.float16:
                scaler.scale(loss).backward()
                if (i + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            pbar.set_postfix({"L_CE": f"{loss_ce.item():.4f}", "L_COS": f"{loss_cosine.item():.4f}"})

        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"rosetta_v6_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "rosetta_v6_final.pt")
    print("🏁 V6 Robust Training Complete!")

if __name__ == "__main__":
    train()
