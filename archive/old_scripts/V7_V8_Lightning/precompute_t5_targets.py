import os
import sys
import torch
from transformers import T5EncoderModel, T5TokenizerFast
from tqdm import tqdm
import glob

# Configuration du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def precompute_t5_targets():
    device = torch.device("cuda")
    
    # 1. Modèle Teacher (On n'utilise que l'encodeur pour extraire le sens)
    print("🎓 Loading T5 Teacher Encoder...")
    model = T5EncoderModel.from_pretrained("t5-small").to(device).half()
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model.eval()

    # 2. Dossiers source
    input_folders = [
        'data/surgical_t5_chunks',
        'data/surgical_monster_chunks',
        'data/manifold_gold'
    ]
    
    output_dir = 'data/t5_target_vectors'
    os.makedirs(output_dir, exist_ok=True)

    # Récupération de tous les fichiers .pt
    all_files = []
    for folder in input_folders:
        all_files.extend(sorted(glob.glob(os.path.join(folder, "*.pt"))))

    print(f"🚀 Found {len(all_files)} chunks to process.")

    for f_path in tqdm(all_files, desc="⚡ Distilling Knowledge"):
        # On charge le batch (BGE, Labels)
        data = torch.load(f_path, map_location='cpu', weights_only=True)
        bge_vecs = data['bge'].half()
        labels = data['labels'].long()
        
        # Le nom du fichier de sortie
        base_name = os.path.basename(f_path)
        save_path = os.path.join(output_dir, f"target_{base_name}")
        
        if os.path.exists(save_path): continue

        # On doit transformer les IDs de tokens en texte pour T5
        # (Ou utiliser l'embedding layer directement, mais passer par le texte est plus sûr pour les poids pré-entraînés)
        phrases = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        t5_targets = []
        batch_size = 256 # Batch pour le Teacher
        
        for i in range(0, len(phrases), batch_size):
            batch_text = phrases[i : i + batch_size]
            with torch.no_grad():
                encoded = tokenizer(batch_text, padding='max_length', truncation=True, max_length=16, return_tensors='pt').to(device)
                # On récupère les états cachés de la dernière couche de l'encodeur
                outputs = model(**encoded)
                hidden_states = outputs.last_hidden_state # [Batch, 16, 512]
                t5_targets.append(hidden_states.cpu().half())

        # Sauvegarde du nouveau chunk distillé
        torch.save({
            'bge': bge_vecs,
            't5_targets': torch.cat(t5_targets)
        }, save_path)

    print(f"🏁 Distillation terminée ! Les cibles T5 sont dans {output_dir}")

if __name__ == "__main__":
    precompute_t5_targets()
