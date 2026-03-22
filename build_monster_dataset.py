import os
import torch
import sentencepiece as spm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

def build_monster():
    parser = argparse.ArgumentParser(description="🚜 Rosetta Holy Trinity Dataset Builder")
    parser.add_argument('--output-dir', type=str, default='data/monster_chunks')
    parser.add_argument('--limit', type=int, default=20000000, help="Nombre total d'exemples à moissonner")
    parser.add_argument('--chunk-size', type=int, default=500000, help="Taille d'un fichier .pt")
    parser.add_argument('--batch-size', type=int, default=1024) # Batch massif pour GPU performant
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Ressources
    print("🛰️ Loading BGE Encoder...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    encoder.half() # On passe l'encodeur en FP16 pour gagner de la vitesse et VRAM
    
    print("👄 Loading Tokenizer...")
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    
    # 2. Liste des Datasets (Holy Trinity)
    dataset_configs = [
        {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "key": "text"},
        {"path": "wikitext", "name": "wikitext-103-v1", "split": "train", "key": "text"},
        # On peut en ajouter d'autres ici si besoin (FineEdu etc.)
    ]

    print(f"🌊 Opening Holy Trinity Streams...")
    
    # Buffers pour le chunk actuel
    all_embeddings = []
    all_token_ids = []
    
    current_chunk_count = 0
    total_count = 0
    chunk_idx = 1
    
    batch_texts = []
    batch_tokens = []

    pbar = tqdm(total=args.limit, desc="🌾 Harvesting BGE Gems")

    # On boucle sur les sources
    for config in dataset_configs:
        if total_count >= args.limit: break
        
        print(f"\n🚜 Stream source: {config['path']}...")
        ds = load_dataset(config['path'], name=config['name'], split=config['split'], streaming=True)

        for example in ds:
            if total_count >= args.limit: break
            
            text = example[config['key']]
            if len(text) < 50: continue # On saute les phrases trop courtes
            
            tokens = sp.encode(text, out_type=int)
            seq_len = 16
            
            # Découpe en segments de 16 tokens
            for i in range(0, len(tokens) - seq_len, 8): # Stride 8 pour recouvrement
                segment_tokens = tokens[i : i + seq_len]
                segment_text = sp.decode(segment_tokens)
                
                batch_texts.append(segment_text)
                batch_tokens.append(torch.tensor(segment_tokens, dtype=torch.int16)) 
                
                if len(batch_texts) >= args.batch_size:
                    with torch.no_grad():
                        embs = encoder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
                        all_embeddings.append(embs.half().cpu())
                        all_token_ids.append(torch.stack(batch_tokens))
                    
                    count = len(batch_texts)
                    total_count += count
                    current_chunk_count += count
                    pbar.update(count)
                    
                    batch_texts = []
                    batch_tokens = []
                    
                    if current_chunk_count >= args.chunk_size:
                        save_path = os.path.join(args.output_dir, f"monster_v5_part_{chunk_idx}.pt")
                        torch.save({
                            'embeddings': torch.cat(all_embeddings),
                            'token_ids': torch.cat(all_token_ids)
                        }, save_path)
                        
                        all_embeddings = []
                        all_token_ids = []
                        current_chunk_count = 0
                        chunk_idx += 1
                        
                if total_count >= args.limit:
                    break

    # Final save
    if all_embeddings:
        save_path = os.path.join(args.output_dir, f"monster_v5_part_{chunk_idx}.pt")
        torch.save({
            'embeddings': torch.cat(all_embeddings),
            'token_ids': torch.cat(all_token_ids)
        }, save_path)

    pbar.close()
    print(f"\n✅ Moisson Trinity terminée ! {total_count} exemples stockés.")

if __name__ == "__main__":
    build_monster()
