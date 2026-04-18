import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import random

# Configuration du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v6 import RosettaV6

class MonsterV7Dataset(Dataset):
    def __init__(self, folders):
        self.samples = []
        for folder in folders:
            files = sorted(glob.glob(os.path.join(folder, "*.pt")))
            if not files:
                print(f"⚠️ Warning: No files found in {folder}")
                continue
            print(f"📦 Loading chunks from {folder}...")
            self.samples.extend(files)
        
        self.all_bge = []
        self.all_labels = []
        
        for f in tqdm(self.samples, desc="🚀 Loading Data to RAM"):
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'].half())
            # On s'assure que les labels ont la même largeur
            labels = data['labels'].to(torch.int16)
            if labels.size(1) != 25:
                # Ajustement dynamique de la largeur si nécessaire
                new_labels = torch.zeros((labels.size(0), 25), dtype=torch.int16)
                width = min(labels.size(1), 25)
                new_labels[:, :width] = labels[:, :width]
                labels = new_labels
            self.all_labels.append(labels)
            
        self.bge = torch.cat(self.all_bge)
        self.labels = torch.cat(self.all_labels).long()
        print(f"✅ V7 Dataset Ready: {len(self.bge)} concepts loaded.")

    def __len__(self): return len(self.bge)
    def __getitem__(self, idx): return self.bge[idx], self.labels[idx]

def train_v7():
    device = torch.device("cuda")
    folders = [
        'data/surgical_t5_chunks',
        'data/surgical_monster_chunks',
        'data/manifold_gold'
    ]
    
    dataset = MonsterV7Dataset(folders)
    # On réduit un peu le batch size car le dataset est immense
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    model = RosettaV6(num_guides=16).to(device)
    
    checkpoint_path = 'checkpoints/rosetta_v6_epoch_25.pt'
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from {checkpoint_path}...")
        sd = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()})

    model = torch.compile(model)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(loader) * 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    scaler = GradScaler()
    cosine_loss_fn = nn.CosineEmbeddingLoss()

    print(f"🔥 Starting OMNISCIENT TRAINING (V7) | Steps: {total_steps}")

    model.train()
    for epoch in range(10):
        pbar = tqdm(loader, desc=f"V7 Epoch {epoch+1}/10")
        for i, (bge_embs, target_ids) in enumerate(pbar):
            bge_embs, target_ids = bge_embs.to(device).float(), target_ids.to(device)
            target_ids[target_ids == 0] = -100 

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ce, _, bge_recon = model(bge_embs, target_ids)
                target = torch.ones(bge_embs.size(0)).to(device)
                loss_cosine = cosine_loss_fn(bge_recon, bge_embs, target)
                loss = loss_ce + 5.0 * loss_cosine

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if i % 100 == 0:
                pbar.set_postfix({"L_CE": f"{loss_ce.item():.4f}", "L_COS": f"{loss_cosine.item():.4f}"})

        # Sauvegarde
        torch.save(model.state_dict(), f"checkpoints/rosetta_v7_epoch_{epoch+1}.pt")

    print("🏁 V7 OMNISCIENT TRAINING COMPLETE!")

if __name__ == "__main__":
    train_v7()
