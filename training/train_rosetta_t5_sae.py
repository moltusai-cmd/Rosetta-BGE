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

from archive.old_scripts.model_sae import SparseAutoencoder
from archive.old_scripts.model_sae_t5 import RosettaSAE2T5

class SAET5Dataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.all_bge = []
        self.all_labels = []
        print("📦 Loading Surgical Fragments for SAE-T5...")
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
    parser = argparse.ArgumentParser(description="⚡ Rosetta SAE-to-T5 Trainer")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--grad-accum', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Charger le SAE pré-entraîné (FROZEN)
    print("🧠 Loading Pre-trained SAE...")
    sae = SparseAutoencoder(d_model=384, d_latent=16384).to(device)
    sae.load_state_dict(torch.load("rosetta_sae_16k.pt", map_location=device))
    sae.eval() # Le SAE est un extracteur de features fixe
    for param in sae.parameters():
        param.requires_grad = False

    # 2. Charger le modèle Rosetta SAE-T5
    print("👄 Initializing Rosetta SAE-T5...")
    model = RosettaSAE2T5(sae_dim=16384).to(device)
    model = torch.compile(model)

    dataset = SAET5Dataset('data/surgical_t5_chunks')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = GradScaler(enabled=not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"🔥 Training Converter | Epochs: {args.epochs} | BF16: {use_bf16}")

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            # A. Passer par le SAE pour obtenir les concepts (Activations)
            with torch.no_grad():
                # Normalisation identique à l'entraînement du SAE
                bge_norm = F.normalize(bge_embs, p=2, dim=1)
                sae_latents = sae.encode(bge_norm) # [Batch, 16384]

            # B. Entraîner le Converter + T5
            with autocast(device_type='cuda', dtype=dtype):
                loss, _ = model(sae_latents, target_ids)
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

        # Sauvegarde
        torch.save(model.state_dict(), f"rosetta_sae_t5_latest.pt")

    print("🏁 SAE-T5 Training Complete!")

if __name__ == "__main__":
    train()
