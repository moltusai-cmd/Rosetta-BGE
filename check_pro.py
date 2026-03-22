import torch
from model import DiffusionRosetta
import os

def check_pro_size():
    vocab_size = 16384
    # d_model=1024, 2 shared layers
    model = DiffusionRosetta(vocab_size=vocab_size, d_model=1024, num_cycles=6)
    
    path = "rosetta_pro_70M.pt"
    torch.save({'model_state_dict': model.state_dict()}, path)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Rosetta-PRO 70M Created!")
    print(f"📊 Parameters: {params/1e6:.2f} M")
    print(f"💾 Disk Size (FP32): {size_mb:.2f} MB")
    print(f"💾 Expected Size (FP16): {size_mb/2:.2f} MB")

if __name__ == "__main__":
    check_pro_size()
