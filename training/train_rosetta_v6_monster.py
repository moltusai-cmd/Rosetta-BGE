import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import random
import threading
import queue

# Configuration du chemin racine pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v6_pro import RosettaV6Pro

def train_rosetta_v6_pro_monster():
    device = torch.device("cuda")
    
    # Cocktail total (4M de points)
    folders = [
        'data/surgical_t5_chunks',
        'data/surgical_monster_chunks',
        'data/manifold_gold'
    ]
    
    files = []
    for folder in folders:
        files.extend(sorted(glob.glob(os.path.join(folder, "*.pt"))))
    
    if not files:
        print("❌ Error: No data found!")
        return

    # 1. Pipeline Asynchrone
    load_queue = queue.Queue(maxsize=2)

    def loader_thread_func():
        local_files = list(files)
        while True:
            random.shuffle(local_files)
            for f in local_files:
                try:
                    data = torch.load(f, map_location='cpu', weights_only=True)
                    labels = data['labels'].to(torch.int16)
                    if labels.size(1) != 25:
                        new_labels = torch.zeros((labels.size(0), 25), dtype=torch.int16)
                        width = min(labels.size(1), 25)
                        new_labels[:, :width] = labels[:, :width]
                        labels = new_labels
                    load_queue.put((data['bge'].half(), labels))
                except Exception as e:
                    print(f"⚠️ Loader error: {e}")

    threading.Thread(target=loader_thread_func, daemon=True).start()

    # 2. Modèle V6 Pro
    model = RosettaV6Pro(num_guides=16).to(device)
    model = torch.compile(model)
    
    # 3. Calcul précis des steps pour OneCycleLR (30 Époques)
    sample_data = torch.load(files[0], map_location='cpu', weights_only=True)
    samples_per_chunk = sample_data['bge'].size(0)
    batch_size = 256
    steps_per_chunk = samples_per_chunk // batch_size
    total_steps = len(files) * steps_per_chunk * 30 # 30 époques
    
    # Optimisation Turbo
    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=8e-4, 
        total_steps=total_steps + 100,
        pct_start=0.15, # Warmup réduit pour un run long
        cycle_momentum=False
    )
    
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"🔥 V6 PRO OVERNIGHT FORGE 🔥")
    print(f"Projector: 10 Blocks / 1024 Dim | 30 Epochs | Steps: {total_steps}")

    model.train()
    for epoch in range(30):
        total_pbar = tqdm(total=len(files), desc=f"V6 Pro Epoch {epoch+1}/30")
        
        for _ in range(len(files)):
            bge_all, labels_all = load_queue.get()
            indices = torch.randperm(len(bge_all))
            
            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                b_bge = bge_all[idx].to(device).float()
                b_labels = labels_all[idx].to(device).long()
                b_labels[b_labels == 0] = -100

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Uniquement la Cross-Entropy
                    loss, _, _ = model(b_bge, b_labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_pbar.set_postfix({"L_CE": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
            total_pbar.update(1)

        # Sauvegarde
        torch.save(model.state_dict(), f"checkpoints/rosetta_v6_pro_monster_e{epoch+1}.pt")

    print("🏁 V6 PRO MONSTER TRAINING COMPLETE.")
    torch.save(model.state_dict(), "checkpoints/rosetta_v6_pro_master.pt")

if __name__ == "__main__":
    train_rosetta_v6_pro_monster()
