import torch
from model import DiffusionRosetta
import os

def check_size():
    # Configuration Mini : d_model=256, n_layers=6
    vocab_size = 16384
    model = DiffusionRosetta(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    
    # Simulation d'un checkpoint
    path = "rosetta_mini_v5.pt"
    torch.save({'model_state_dict': model.state_dict()}, path)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Modèle Mini créé !")
    print(f"📊 Paramètres : {params/1e6:.2f} M")
    print(f"💾 Taille sur disque : {size_mb:.2f} Mo")

if __name__ == "__main__":
    check_size()
