import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import sentencepiece as spm
from sentence_transformers import SentenceTransformer
import argparse
import glob
import random

# Import Rosetta Diffusion from local model.py
from model import DiffusionRosetta
from model_v6 import DiffusionRosettaV6

def train_monster():
    parser = argparse.ArgumentParser(description="🚜 Rosetta-Monster 20M Training Suite")
    parser.add_argument('--data-dir', type=str, default='/home/nini/Model_training/data/monster_chunks')
    parser.add_argument('--v6', action='store_true', help="Utiliser la version v6 (Ultra)")
    parser.add_argument('--lr', type=float, default=5e-4) 
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--sem-weight', type=float, default=0.5)
    parser.add_argument('--pct-start', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=512) 
    parser.add_argument('--grad-accum', type=int, default=2)   # 512 * 2 = 1024 (Effective)
    parser.add_argument('--epochs', type=int, default=6) 
    parser.add_argument('--resume', type=str, default=None, help="Chemin vers le checkpoint .pt")
    parser.add_argument('--trial', action='store_true', help="Mode Auto-ML : pas de load/save")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    vocab_size = sp.get_piece_size()
    mask_id = vocab_size

    # 1. Scanning Chunks
    chunk_files = sorted(glob.glob(os.path.join(args.data_dir, "monster_v5_part_*.pt")))
    print(f"📦 Found {len(chunk_files)} chunks in {args.data_dir}")
    
    # 2. Forge du PRO Rosetta
    if args.v6:
        print("🏗️ Forge du Dénoiseur ULTRA 70M v6.0 (SwiGLU, Weight Tying)...")
        model = DiffusionRosettaV6(
            vocab_size=vocab_size, 
            d_model=1024, 
            n_heads=16,
            num_cycles=6
        ).to(device)
    else:
        print("🏗️ Forge du Dénoiseur PRO 70M v5.2 (Original)...")
        model = DiffusionRosetta(
            vocab_size=vocab_size, 
            d_model=1024, 
            n_heads=16,
            num_cycles=6
        ).to(device)

    # 🛰️ BGE Encoder (pour Short-Form Augmentation)
    print("🛰️ Loading BGE Encoder for on-the-fly augmentation...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    start_epoch = 0
    start_chunk = 0
    global_step = 0
    if not args.trial and args.resume and os.path.exists(args.resume):
        print(f"🔄 Reprise depuis {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        global_step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)
        start_chunk = ckpt.get('chunk_idx', -1) + 1
        print(f"   ✅ Repris au step {global_step} (Epoch {start_epoch + 1}, Chunk {start_chunk + 1})")

    # 🔥 JIT Compilation (Standard) - Moins de VRAM
    print("⚡ Compiling model kernels (standard)...")
    model = torch.compile(model)

    # Estimation réelle des steps (basée sur vos logs : 1008 it / 2 accum = 504 steps par chunk)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_chunk = 504 
    steps_per_epoch = steps_per_chunk * len(chunk_files)
    
    # On utilise l'argument passé en ligne de commande
    total_steps = steps_per_epoch * args.epochs
    
    # Injection manuelle des paramètres pour OneCycleLR si on reprend
    if global_step > 0:
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = args.lr / 25.0
            if 'max_lr' not in param_group:
                param_group['max_lr'] = args.lr
            if 'min_lr' not in param_group:
                param_group['min_lr'] = args.lr / 10000.0

    # On bride le LR max pour la sécurité de l'architecture v6
    safe_lr = min(args.lr, 8e-4)
    if args.lr > 8e-4:
        print(f"⚠️ LR {args.lr} est trop risqué pour v6. Bridage de sécurité à {safe_lr}")

    print(f"📈 Scheduler: Reprise au step {global_step} sur {total_steps} (Fin à l'Epoch {args.epochs})")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=safe_lr, 
        total_steps=total_steps,
        last_epoch=global_step-1,
        pct_start=args.pct_start,
        cycle_momentum=False
    )

    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # SWA (Stochastic Weight Averaging) pour la fin
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(args.epochs * 0.8) # On commence SWA aux derniers 20%
    
    print(f"🌊 Entraînement sur {total_steps} steps (Effective batch size: {effective_batch})...")
    model.train()

    optimizer.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        # On ne mélange les chunks que si on commence une nouvelle epoch
        if start_chunk == 0:
            random.shuffle(chunk_files) 
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            # Sauter les chunks déjà faits lors d'une reprise
            if epoch == start_epoch and chunk_idx < start_chunk:
                continue
            data = torch.load(chunk_path, map_location="cpu", weights_only=True)
            dataset = TensorDataset(data['embeddings'], data['token_ids'].long())
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1} | Chunk {chunk_idx+1}/{len(chunk_files)}")
            
            for i, (embs, tokens) in enumerate(pbar):
                embs, tokens = embs.to(device).float(), tokens.to(device)
                batch_size = tokens.size(0)

                # --- 🛰️ SUB-SEGMENT AUGMENTATION (10% de chance) ---
                if random.random() < 0.10:
                    new_texts = []
                    new_tokens = torch.full_like(tokens, sp.eos_id())
                    for b in range(batch_size):
                        # Fenêtre glissante : on choisit un segment aléatoire [i:j]
                        i = random.randint(0, 15)
                        j = random.randint(i, 15)
                        
                        sub_segment = tokens[b, i:j+1]
                        length = len(sub_segment)
                        
                        # On replace le segment au début (position 0) pour Rosetta
                        new_tokens[b, :length] = sub_segment
                        new_texts.append(sp.decode(sub_segment.tolist()))
                    
                    tokens = new_tokens # On remplace les targets par les versions courtes
                    
                    # On recalcule l'embedding pour ce segment précis
                    with torch.no_grad():
                        embs = encoder.encode(new_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).float()
                # --------------------------------------------------
                
                # --- 🎭 CURRICULUM MASKING ---
                # On augmente la difficulté (t_max) avec le temps
                progress = global_step / total_steps
                t_max = 0.5 + (0.5 * progress) # De 0.5 à 1.0
                t = (torch.rand((batch_size, 1), device=device) * t_max)
                
                noise_mask = (torch.rand(tokens.shape, device=device) < t).long()
                input_tokens = tokens * (1 - noise_mask) + mask_id * noise_mask
                
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, semantic_pred = model(embs, input_tokens, return_semantic=True) 
                    
                    # 1. Cross-Entropy (Orthographe/Syntaxe)
                    loss_ce = criterion(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
                    
                    # 2. InfoNCE Sémantique (Contrastive)
                    # --- RÉGIME DE RÉSOLUTION DYNAMIQUE ---
                    if epoch < 3:
                        temp_contrast = 0.1 # Dégrossissage
                    elif epoch < 6:
                        temp_contrast = 0.07 # Polissage HD
                    else:
                        temp_contrast = 0.05 # Polissage DIAMANT (Dès Epoch 7)
                        
                    sim_matrix = F.cosine_similarity(semantic_pred.unsqueeze(1), embs.unsqueeze(0), dim=-1) / temp_contrast
                    labels = torch.arange(batch_size, device=device)
                    loss_sem = F.cross_entropy(sim_matrix, labels)
                    
                    # Normalisation par grad_accum
                    loss = (loss_ce + args.sem_weight * loss_sem) / args.grad_accum
                
                scaler.scale(loss).backward()
                
                if (i + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    
                    # Clipping dynamique corrélé à la résolution
                    if epoch < 3:
                        current_clip = 1.0
                    elif epoch < 6:
                        current_clip = 0.5
                    else:
                        current_clip = 0.3 # Sécurité maximale en mode diamant
                        
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    
                    # Mise à jour SWA en fin d'entraînement
                    if epoch >= swa_start:
                        swa_model.update_parameters(model)

                    if global_step % 100 == 0:
                        preds = torch.argmax(logits, dim=-1)
                        acc = (preds == tokens).float().mean()
                        metrics_str = f"STEP={global_step} L_CE={loss_ce.item() * args.grad_accum:.4f} L_SEM={loss_sem.item():.4f} Acc={acc.item():.4f} GN={grad_norm.item():.2f}"
                        if args.trial:
                            print(metrics_str) # Nouvelle ligne pour le tuner
                        
                        pbar.set_postfix({
                            'L_CE': f'{loss_ce.item() * args.grad_accum:.2f}', 
                            'L_SEM': f'{loss_sem.item():.2f}', 
                            'Acc': f'{acc.item():.2%}',
                            'GN': f'{grad_norm.item():.2f}',
                            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
                        })

            # Sauvegarde intermédiaire
            if not args.trial and (chunk_idx + 1) % 5 == 0:
                os.makedirs("checkpoints_monster", exist_ok=True)
                save_path = "rosetta_mini_monster_v5.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'step': global_step,
                    'epoch': epoch,
                    'chunk_idx': chunk_idx
                }, save_path)
                
                # Test de décodage
                model.eval()
                with torch.no_grad():
                    gen_tokens = model.decode(embs[0:1], num_steps=16)
                    print(f"\n🔮 Sample: {sp.decode(gen_tokens[0].tolist())}")
                    print(f"🎯 Target: {sp.decode(tokens[0].tolist())}")
                model.train()

        # Sauvegarde OBLIGATOIRE en fin d'epoch
        if not args.trial:
            start_chunk = 0 # Réinitialisation pour l'epoch suivante
            save_path = f"rosetta_mini_monster_epoch_{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'step': global_step,
                'epoch': epoch + 1, # Prêt pour l'epoch suivante
                'chunk_idx': -1
            }, save_path)
            # On met aussi à jour le checkpoint principal
            torch.save(torch.load(save_path), "rosetta_mini_monster_v5.pt")
            print(f"💾 Epoch {epoch+1} terminée et sauvegardée.")

            # Sauvegarde finale SWA
            print("💎 Finalisation : Sauvegarde du modèle SWA (Stochastic Weight Averaging)...")
            torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'step': global_step,
            'epoch': args.epochs
            }, "rosetta_mini_monster_v6_final_swa.pt")

            print("\n✅ Entraînement MONSTER terminé ! Rosetta-Mini est maintenant un génie.")
if __name__ == "__main__":
    train_monster()
