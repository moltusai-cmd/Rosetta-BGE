import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import sentencepiece as spm
import argparse

# Import Rosetta Diffusion from local model.py
from model import DiffusionRosetta

def train_mini():
    parser = argparse.ArgumentParser(description="🌀 Rosetta-Mini TURBO (Latent Pre-trained)")
    parser.add_argument('--dataset', type=str, default='/home/nini/Model_training/data/latent_dataset_v3_16tokens.pt')
    parser.add_argument('--lr', type=float, default=4e-4) # Plus haut pour un petit modèle sur dataset fixe
    parser.add_argument('--batch-size', type=int, default=512) # On peut monter le batch car plus de modèle BGE
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    vocab_size = sp.get_piece_size()
    mask_id = vocab_size

    # 1. Chargement du dataset en RAM (Turbo Mode)
    print(f"📦 Chargement du dataset latent : {args.dataset}...")
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    embeddings = data['embeddings'] # [1.5M, 384]
    token_ids = data['token_ids']   # [1.5M, 16]
    
    dataset = TensorDataset(embeddings, token_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"📊 Dataset prêt : {len(dataset)} exemples.")

    # 2. Forge du MINI Rosetta
    print("🏗️ Forge du Dénoiseur MINI (d_model=256, n_layers=6)...")
    model = DiffusionRosetta(
        vocab_size=vocab_size, 
        d_model=256, 
        n_layers=6, 
        n_heads=8
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=len(loader), 
        epochs=args.epochs
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    print("🌊 Début de l'entraînement TURBO. On cristallise le sens...")
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for embs, tokens in pbar:
            embs, tokens = embs.to(device), tokens.to(device)
            batch_size = tokens.size(0)
            
            # --- LOGIQUE DE DIFFUSION (Masking) ---
            t = torch.rand((batch_size, 1), device=device)
            noise_mask = (torch.rand(tokens.shape, device=device) < t).long()
            input_tokens = tokens * (1 - noise_mask) + mask_id * noise_mask
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(embs, input_tokens) # [B, 16, Vocab]
                loss = criterion(logits.reshape(-1, vocab_size), tokens.reshape(-1))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            if pbar.n % 50 == 0:
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == tokens).float().mean()
                pbar.set_postfix({'L': f'{loss.item():.3f}', 'Acc': f'{acc.item():.2%}'})

        # Sauvegarde après chaque epoch
        os.makedirs("checkpoints_mini", exist_ok=True)
        save_path = f"rosetta_mini_v5_epoch_{epoch+1}.pt"
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, save_path)
        
        # Test rapide sur le premier exemple du batch
        model.eval()
        with torch.no_grad():
            gen_tokens = model.decode(embs[0:1], num_steps=16)
            print(f"\n🔮 Sample Output : {sp.decode(gen_tokens[0].tolist())}")
            print(f"🎯 Target        : {sp.decode(tokens[0].tolist())}")
        model.train()

    print("\n✅ Entraînement terminé ! Le Rosetta-Mini est prêt.")

if __name__ == "__main__":
    train_mini()
