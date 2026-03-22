import torch
from model import DiffusionRosetta
import os

def convert_mini_to_fp16(input_path="rosetta_mini_v5.pt", output_path="rosetta_mini_fp16.pt"):
    device = "cpu"
    vocab_size = 16384
    
    # 1. Re-créer l'architecture Mini
    model = DiffusionRosetta(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    
    # 2. Charger les poids (s'ils existent)
    if os.path.exists(input_path):
        ckpt = torch.load(input_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ Poids chargés depuis {input_path}")
    else:
        print("⚠️ Aucun poids trouvé, création d'un modèle FP16 vierge pour test de taille.")

    # 3. Convertir en FP16
    model = model.half() 
    
    # 4. Sauvegarder
    torch.save({'model_state_dict': model.state_dict()}, output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"🚀 Modèle converti en FP16 !")
    print(f"💾 Taille finale : {size_mb:.2f} Mo")
    print(f"📍 Sauvegardé sous : {output_path}")

if __name__ == "__main__":
    convert_mini_to_fp16()
