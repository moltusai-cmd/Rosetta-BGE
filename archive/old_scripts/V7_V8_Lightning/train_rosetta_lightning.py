import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import random
import threading
import queue

# Configuration du chemin racine pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_v7 import RosettaV7

def train_lightning_v7_contrast():
    device = torch.device("cuda")
    data_dir = 'data/t5_target_vectors'
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    
    if not files:
        print(f"❌ Error: No files found in {data_dir}.")
        return

    load_queue = queue.Queue(maxsize=2)

    def loader_thread_func():
        local_files = list(files)
        while True:
            random.shuffle(local_files)
            for f in local_files:
                try:
                    data = torch.load(f, map_location='cpu', weights_only=True)
                    load_queue.put((data['bge'], data['t5_targets']))
                except Exception as e:
                    print(f"⚠️ Loader error: {e}")

    threading.Thread(target=loader_thread_func, daemon=True).start()

    # 1. Modèle V7.2 (The Discriminator)
    model = RosettaV7(num_guides=16).to(device)
    model.float()
    model.t5.half() 
    for param in model.t5.parameters():
        param.requires_grad = False
    
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()

    print(f"🚀 GIGA-LIGHTNING V7.2 (CONTRASTIVE) | Device: {device} | Batch: 1024")

    for epoch in range(30):
        total_pbar = tqdm(total=len(files), desc=f"V7.2 Epoch {epoch+1}/30")
        
        for _ in range(len(files)):
            bge_raw, t5_raw = load_queue.get()
            bge_all = bge_raw.to(device).float()
            t5_all = t5_raw.to(device).float()
            
            batch_size = 1024
            for i in range(0, len(bge_all), batch_size):
                bge_batch = bge_all[i : i + batch_size]
                t5_batch = t5_all[i : i + batch_size]

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Forward
                    x = model.input_expander(bge_batch)
                    x = model.input_norm(x)
                    x = model.brain(x)
                    guides = model.output_projector(x).view(-1, 16, model.d_model)
                    
                    # --- LOSS MULTI-OBJECTIVE ---
                    
                    # 🅰️ Loss MSE (Attraction brute)
                    loss_mse = F.mse_loss(guides, t5_batch)
                    
                    # 🅱️ Loss Cosine (Alignement de direction)
                    guides_flat = guides.view(-1, 512)
                    t5_flat = t5_batch.view(-1, 512)
                    loss_cos = 1.0 - F.cosine_similarity(guides_flat, t5_flat).mean()
                    
                    # ⚡ Loss Contrastive (Discrimination / Répulsion)
                    # On compare les "résumés" de chaque fragment dans le batch
                    pred_pool = guides.mean(dim=1)
                    target_pool = t5_batch.mean(dim=1)
                    # Matrice de logits (Similarité croisée)
                    sim_matrix = torch.matmul(F.normalize(pred_pool, dim=-1), 
                                            F.normalize(target_pool, dim=-1).t())
                    # La température (20.0) force le modèle à être très sûr de lui
                    contrast_labels = torch.arange(sim_matrix.size(0)).to(device)
                    loss_contrast = F.cross_entropy(sim_matrix * 20.0, contrast_labels)
                    
                    # Fusion des pertes
                    loss = loss_mse + 5.0 * loss_cos + 0.1 * loss_contrast

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_pbar.set_postfix({
                "MSE": f"{loss_mse.item():.4f}", 
                "COS": f"{loss_cos.item():.4f}",
                "CTR": f"{loss_contrast.item():.4f}"
            })
            total_pbar.update(1)

        # Sauvegarde
        torch.save(model.state_dict(), f"checkpoints/rosetta_v7_ultra_latest.pt")

    print("🏁 V7.2 CONTRASTIVE TRAINING COMPLETE!")

if __name__ == "__main__":
    train_lightning_v7_contrast()
