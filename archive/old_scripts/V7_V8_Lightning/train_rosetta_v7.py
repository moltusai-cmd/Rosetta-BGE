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
import threading
import queue

# Configuration du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v7 import RosettaV7

def train_rosetta_v7_ultra():
    device = torch.device("cuda")
    
    # Dossiers de données (Le cocktail complet)
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

    # 1. Pipeline Asynchrone (Pour saturer la 5080)
    load_queue = queue.Queue(maxsize=2)

    def loader_thread_func():
        local_files = list(files)
        while True:
            random.shuffle(local_files)
            for f in local_files:
                try:
                    data = torch.load(f, map_location='cpu', weights_only=True)
                    # On pré-formate les labels en 25 tokens pour la stabilité
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

    # 2. Modèle V7 Ultra
    model = RosettaV7(num_guides=16).to(device)
    
    # Reprise de la V6 si possible pour la stabilité des poids T5
    checkpoint_v6 = 'checkpoints/rosetta_v6_epoch_25.pt'
    if os.path.exists(checkpoint_v6):
        print(f"🔄 Hydrating T5 from {checkpoint_v6}...")
        sd = torch.load(checkpoint_v6, map_location=device, weights_only=False)
        # On ne charge QUE T5 car le projecteur a changé
        t5_sd = {k.replace('t5.', ''): v for k, v in sd.items() if k.startswith('t5.') or k.startswith('_orig_mod.t5.')}
        model.t5.load_state_dict(t5_sd, strict=False)

    model = torch.compile(model)
    
    # 3. Optimisation
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(files) * 20 # 20 époques (On a le temps !)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    scaler = GradScaler()
    cosine_loss_fn = nn.CosineEmbeddingLoss()

    print(f"🔥 FINAL OMNISCIENT FORGE (V7 ULTRA) 🔥")
    print(f"Dataset: {len(files)} chunks | Device: {device}")

    for epoch in range(20):
        total_pbar = tqdm(total=len(files), desc=f"V7 Ultra Epoch {epoch+1}/20")
        
        for _ in range(len(files)):
            bge_all, labels_all = load_queue.get()
            
            # Entraînement par mini-batch de 256
            batch_size = 256
            indices = torch.randperm(len(bge_all))
            
            model.train()
            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                b_bge = bge_all[idx].to(device).float()
                b_labels = labels_all[idx].to(device).long()
                
                # Masquage des tokens de padding (0 -> -100 pour T5)
                b_labels[b_labels == 0] = -100

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Le Forward de la V7 Ultra fait tout : Proj -> T5 -> Mirror
                    loss_ce, _, bge_recon = model(b_bge, b_labels)
                    
                    # Perte de fidélité sémantique (Le Miroir)
                    cos_target = torch.ones(b_bge.size(0)).to(device)
                    loss_cosine = cosine_loss_fn(bge_recon, b_bge, cos_target)
                    
                    # Poids : Priorité au texte (CE) avec un fort guidage sémantique (COS)
                    total_loss = loss_ce + 5.0 * loss_cosine

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_pbar.set_postfix({"CE": f"{loss_ce.item():.3f}", "COS": f"{loss_cosine.item():.4f}"})
            total_pbar.update(1)

        # Sauvegarde toutes les 2 époques
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/rosetta_v7_ultra_final_e{epoch+1}.pt")

    print("🏁 THE OMNISCIENT V7 IS COMPLETE.")
    torch.save(model.state_dict(), "checkpoints/rosetta_v7_ultra_master.pt")

if __name__ == "__main__":
    train_rosetta_v7_ultra()
