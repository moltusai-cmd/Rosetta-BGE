import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    🧠 SEMANTIC RECTIFIER (SAE)
    - Input: BGE 384d (Dense)
    - Latent: 16384d (Sparse)
    - Goal: Disentangle polysemic concepts
    """
    def __init__(self, d_model=384, d_latent=16384):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent

        # 1. ENCODER (Analyse sémantique)
        # On soustrait un 'bias' avant le ReLU pour forcer la rareté (Sparsity)
        self.encoder = nn.Linear(d_model, d_latent)
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        
        # 2. DECODER (Synthèse sémantique)
        # On utilise des poids liés (ou non) et on normalise les colonnes
        self.decoder = nn.Linear(d_latent, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Initialisation spécifique pour les SAE
        nn.init.orthogonal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.t().clone()

    def encode(self, x):
        # x: [batch, 384]
        x_centered = x - self.b_dec
        latents = F.relu(self.encoder(x_centered) + self.b_enc)
        return latents

    def forward(self, x, l1_coeff=1e-3):
        # 1. Encodage vers l'espace clairsemé
        latents = self.encode(x)
        
        # 2. Reconstruction
        recons = self.decoder(latents) + self.b_dec
        
        # 3. Métriques de rareté
        # Combien de neurones sont allumés en moyenne ?
        sparsity = (latents > 0).float().mean(dim=-1).mean()
        # Perte L1 pour forcer la rareté
        l1_loss = latents.abs().sum(dim=-1).mean()
        
        return recons, l1_loss, sparsity, latents

if __name__ == "__main__":
    model = SparseAutoencoder()
    fake_bge = torch.randn(4, 384)
    recons, l1, sp, _ = model(fake_bge)
    print(f"✅ SAE Prototyped. Latent size: 16384")
    print(f"📊 Initial Sparsity: {sp.item()*100:.2f}%")
