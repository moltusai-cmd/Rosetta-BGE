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

# Configuration du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v8 import RosettaTransformer

def train_rosetta_v8_attention():
    device = torch.device("cuda")
    
    # 4 Millions de points
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
                    print(f"⚠️ Loader error on {f}: {e}")

    threading.Thread(target=loader_thread_func, daemon=True).start()

    # 2. Modèle V8 Attention
    model = RosettaTransformer(num_guides=16).to(device)
    model = torch.compile(model)
    
    # 3. Optimisation
    # On commence avec un LR de 1e-4 pour l'attention
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(files) * 20
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=total_steps)
    
    scaler = GradScaler()
    cosine_loss_fn = nn.CosineEmbeddingLoss()

    print(f"🔥 V8 ATTENTION FORGE ACTIVATED 🔥")
    print(f"Architecture: Transformer Encoder (6 layers, 8 heads)")

    model.train()
    for epoch in range(20):
        total_pbar = tqdm(total=len(files), desc=f"V8 Epoch {epoch+1}/20")
        
        for _ in range(len(files)):
            bge_all, labels_all = load_queue.get()
            
            batch_size = 256
            indices = torch.randperm(len(bge_all))
            
            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                b_bge = bge_all[idx].to(device).float()
                b_labels = labels_all[idx].to(device).long()
                b_labels[b_labels == 0] = -100

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Forward V8 (Transformer)
                    loss_ce, _, bge_recon = model(b_bge, b_labels)
                    
                    # Mirror Head Loss
                    cos_target = torch.ones(b_bge.size(0)).to(device)
                    loss_cosine = cosine_loss_fn(bge_recon, b_bge, cos_target)
                    
                    total_loss = loss_ce + 3.0 * loss_cosine

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_pbar.set_postfix({"L_CE": f"{loss_ce.item():.3f}", "L_COS": f"{loss_cosine.item():.4f}"})
            total_pbar.update(1)

        # Sauvegarde stratégique
        torch.save(model.state_dict(), f"checkpoints/rosetta_v8_attention_e{epoch+1}.pt")

    print("🏁 V8 ATTENTION TRAINING COMPLETE.")
    torch.save(model.state_dict(), "checkpoints/rosetta_v8_master.pt")

if __name__ == "__main__":
    train_rosetta_v8_attention()
