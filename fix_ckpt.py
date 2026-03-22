import torch
import os

def fix_checkpoint():
    ckpt_path = "rosetta_mini_monster_v5.pt"
    if not os.path.exists(ckpt_path):
        print("❌ Checkpoint introuvable.")
        return

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    print(f"Avant - Epoch: {ckpt.get('epoch')}, Chunk: {ckpt.get('chunk_idx')}, Step: {ckpt.get('step')}")
    
    # On force le passage à l'Epoch 3 (index 2)
    # On met le step à un niveau cohérent avec la fin de l'Epoch 2
    # 39 chunks * 1008 steps = ~39312 steps par epoch.
    # Epoch 1 + Epoch 2 = ~78624 steps.
    # On va garder le global_step actuel car le scheduler en dépend
    
    ckpt['epoch'] = 2 # Index 2 = Epoch 3
    ckpt['chunk_idx'] = -1 # Pour reprendre au chunk 0
    
    torch.save(ckpt, ckpt_path)
    print(f"✅ Après - Epoch: {ckpt.get('epoch')}, Chunk: {ckpt.get('chunk_idx')}, Step: {ckpt.get('step')}")
    print("🚀 Checkpoint prêt pour la reprise à l'Epoch 3!")

if __name__ == "__main__":
    fix_checkpoint()
