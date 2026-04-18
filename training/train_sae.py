import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import glob
import argparse

from archive.old_scripts.model_sae import SparseAutoencoder

class BGEVectorDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.all_bge = []
        print(f"📦 Loading BGE Vectors for SAE...")
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.all_bge.append(data['bge'])
        self.bge = torch.cat(self.all_bge).float()
        print(f"✅ Ready! {len(self.bge)} vectors loaded.")

    def __len__(self): return len(self.bge)
    def __getitem__(self, idx): return self.bge[idx]

def train():
    parser = argparse.ArgumentParser(description="🧠 Train Semantic SAE on BGE Vectors")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024) # Batch massif pour SAE
    parser.add_argument('--l1-coeff', type=float, default=0.01) # Force la rareté
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BGEVectorDataset('data/surgical_t5_chunks')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SparseAutoencoder(d_model=384, d_latent=16384).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"🔥 Training Semantic SAE | Device: {device} | L1 Coeff: {args.l1_coeff}")

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for bge in pbar:
            bge = bge.to(device)

            # Normalisation du vecteur BGE (Important pour SAE)
            bge = F.normalize(bge, p=2, dim=1)

            # Forward
            recons, l1_loss, sparsity, _ = model(bge)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recons, bge)
            
            # Perte Totale
            loss = recon_loss + args.l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "L_MSE": f"{recon_loss.item():.6f}", 
                "Sparsity": f"{sparsity.item()*100:.2f}%",
                "Neurons": f"{int(sparsity.item() * 16384)}"
            })

    torch.save(model.state_dict(), "rosetta_sae_16k.pt")
    print("🏁 SAE Training Complete!")

if __name__ == "__main__":
    train()
