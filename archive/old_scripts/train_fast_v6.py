import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import glob

# Import Rosetta Diffusion from local model_v6.py
from core.model_v6 import DiffusionRosettaV6

class PrecomputedDataset(Dataset):
    def __init__(self, data_dir, mask_id=16384):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.mask_id = mask_id
        self.all_embeddings = []
        self.all_token_ids = []
        
        print(f"📦 Loading {len(self.chunk_files)} chunks into memory...")
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_embeddings.append(data['embeddings'])
            self.all_token_ids.append(data['token_ids'])
            
        self.embeddings = torch.cat(self.all_embeddings)
        self.token_ids = torch.cat(self.all_token_ids)
        print(f"✅ Loaded {len(self.token_ids)} segments.")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        target = self.token_ids[idx].clone()
        emb = self.embeddings[idx].clone()
        
        # Diffusion Masking (On the fly)
        x = target.clone()
        mask_prob = torch.rand(1).item() * 0.9 + 0.1 
        mask = torch.rand(x.shape) < mask_prob
        x[mask] = self.mask_id
        
        return emb, x, target

def train():
    parser = argparse.ArgumentParser(description="⚡ Rosetta Fast Training Suite")
    parser.add_argument('--data-dir', type=str, default='data/robust_chunks')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--compile', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | BF16: Enabled | Compile: {args.compile}")

    dataset = PrecomputedDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("🏗️ Forge du Dénoiseur Rosetta V6 (70M)...")
    model = DiffusionRosettaV6(
        vocab_size=16384, 
        d_model=1024, 
        n_heads=16, 
        num_cycles=6,
        num_tokens=16
    ).to(device)
    
    if args.compile:
        print("⚡ Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}. Proceeding without compilation.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Scaler is needed for BF16/FP16 mixed precision
    scaler = GradScaler()

    model.train()
    
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for bge_embs, x, target in pbar:
            bge_embs, x, target = bge_embs.to(device), x.to(device), target.to(device)
            
            # Autocast to BF16 (if available) or FP16
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            with autocast(device_type='cuda', dtype=dtype):
                logits, sem_pred = model(bge_embs, x, return_semantic=True)
                
                # Reconstruction Loss
                loss_ce = criterion(logits.view(-1, 16384 + 1), target.view(-1))
                # Semantic Mirror Loss
                loss_sem = F.mse_loss(sem_pred, bge_embs.float())
                
                loss = loss_ce + 0.5 * loss_sem

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            acc = (logits.argmax(dim=-1) == target).float().mean().item()
            pbar.set_postfix({
                "L_CE": f"{loss_ce.item():.4f}", 
                "L_SEM": f"{loss_sem.item():.4f}", 
                "Acc": f"{acc:.2%}"
            })

    print("✅ Training Complete!")
    torch.save(model.state_dict(), "rosetta_v6_fast.pt")

if __name__ == "__main__":
    train()
