import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import argparse
import random

# Import Rosetta Diffusion from local model.py
from model import DiffusionRosetta

class FineWebDiffusionDataset(IterableDataset):
    def __init__(self, sp_model, seq_len=16):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)
        self.seq_len = seq_len
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    def __iter__(self):
        for example in self.dataset:
            tokens = self.sp.encode(example['text'], out_type=int)
            for i in range(0, len(tokens) - self.seq_len, 4): # Stride 4
                segment = tokens[i : i + self.seq_len]
                segment_text = self.sp.decode(segment)
                yield segment_text, torch.tensor(segment, dtype=torch.long)

def train_diffusion():
    parser = argparse.ArgumentParser(description="🌀 Rosetta Diffusion v5.0")
    parser.add_argument('--sp-model', type=str, default='tokenizer.model')
    parser.add_argument('--lr', type=float, default=2e-4) # Plus bas pour la diffusion
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--resume', type=str, default=None, help="Chemin vers le checkpoint .pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    vocab_size = sp.get_piece_size()
    mask_id = vocab_size

    # 1. BGE Encoder
    print("🚀 Chargement du Nez (BGE-small)...")
    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    # 2. Diffusion Rosetta Model
    print("🏗️ Forge du Dénoiseur (Diffusion Rosetta v5.0)...")
    model = DiffusionRosetta(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    start_step = 0
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Reprise depuis {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                param_group['initial_lr'] = args.lr 
        start_step = ckpt.get('step', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"   ✅ Repris au step {start_step} (Best Acc: {best_acc:.2%})")

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr

    total_steps = args.steps
    if start_step >= total_steps * 0.9:
        total_steps = total_steps * 2

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, last_epoch=start_step-1)
    
    # 3. Data
    dataset = FineWebDiffusionDataset(args.sp_model)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    print("🌊 Début de la Diffusion. On cristallise le flux...")
    pbar = tqdm(total=total_steps, desc="Diffusion Steps", initial=start_step)
    step = start_step
    model.train()

    for texts, tokens in loader:
        if step >= total_steps: break
        with torch.no_grad():
            embs = encoder.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True).float()
        
        tokens = tokens.to(device)
        batch_size = tokens.size(0)
        
        # --- LOGIQUE DE DIFFUSION (Masking) ---
        t = torch.rand((batch_size, 1), device=device)
        mask_prob = t 
        
        noise_mask = (torch.rand(tokens.shape, device=device) < mask_prob).long()
        input_tokens = tokens * (1 - noise_mask) + mask_id * noise_mask
        
        # Forward
        with autocast(device_type='cuda'):
            logits = model(embs, input_tokens) 
            
            # --- POSITIONAL LOSS WEIGHTING ---
            pos_weights = torch.ones(16, device=device)
            pos_weights[:4] = 2.0  
            pos_weights[-4:] = 2.0 
            
            criterion_raw = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
            loss_per_token = criterion_raw(logits.reshape(-1, vocab_size), tokens.reshape(-1))
            loss_per_token = loss_per_token.view(batch_size, 16)
            
            loss = (loss_per_token * pos_weights).mean()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # D. Logging
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == tokens).float().mean()
        
        pbar.update(1)
        pbar.set_postfix({'L': f'{loss.item():.3f}', 'Acc': f'{acc.item():.2%}', 'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'})

        if step % 1000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'best_acc': acc.item()
            }, f"checkpoints/rosetta_v5_step_{step}.pt")
            
            # Test de génération rapide
            model.eval()
            with torch.no_grad():
                sample_emb = embs[0:1]
                gen_tokens = model.decode(sample_emb, num_steps=16) # Use decode method from model.py
                gen_text = sp.decode(gen_tokens[0].tolist())
                print(f"\n🔮 [Step {step}]")
                print(f"Target : {texts[0]}")
                print(f"Output : {gen_text}")
                print("-" * 50)
            model.train()

        step += 1
        if step >= total_steps: break

if __name__ == "__main__":
    train_diffusion()
