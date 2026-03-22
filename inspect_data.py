import torch

def inspect_dataset(path):
    print(f"🧐 Inspection de {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"🔑 Clé '{k}' : Tensor de forme {v.shape}")
                else:
                    print(f"🔑 Clé '{k}' : Type {type(v)}")
        elif isinstance(data, list):
            print(f"📦 Liste de {len(data)} éléments. Premier élément : {type(data[0])}")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    inspect_dataset("/home/nini/Model_training/data/latent_dataset_v3_16tokens.pt")
