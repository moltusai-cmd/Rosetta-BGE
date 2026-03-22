import torch
from model import DiffusionRosetta
import os

def check_recursive_size():
    vocab_size = 16384
    # d_model=512, but only 1 shared layer!
    model = DiffusionRosetta(vocab_size=vocab_size, d_model=512, num_loops=12)
    
    path = "rosetta_recursive_v5.pt"
    torch.save({'model_state_dict': model.state_dict()}, path)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Recursive Rosetta Created!")
    print(f"📊 Parameters: {params/1e6:.2f} M")
    print(f"💾 Disk Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    check_recursive_size()
