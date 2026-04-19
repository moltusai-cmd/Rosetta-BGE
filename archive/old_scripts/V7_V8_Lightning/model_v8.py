import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class RosettaTransformer(nn.Module):
    """
    🛰️ ROSETTA-V8 (Attention Edition)
    - 16 Learnable Latent Queries
    - Deep Transformer Reasoning (6 Layers)
    - Cross-talk between semantic guides
    """
    def __init__(self, bge_dim=384, t5_name="t5-small", num_guides=16, n_layers=6):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.d_model = self.t5.config.d_model # 512
        self.num_guides = num_guides

        # 1. Projection d'entrée (BGE -> Espace Transformer)
        self.bge_proj = nn.Linear(bge_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # 2. Learnable Latent Queries (Les 16 "aspirateurs à sens")
        self.latent_queries = nn.Parameter(torch.randn(num_guides, self.d_model))
        
        # 3. Le Cerveau à Attention (6 couches, 8 têtes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True # Pre-Norm pour la stabilité
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Miroir (Optionnel)
        self.mirror_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, bge_dim)
        )

    def forward(self, bge_emb, target_ids=None):
        batch_size = bge_emb.size(0)
        
        # Transformation du BGE en token de contexte
        bge_token = self.bge_proj(bge_emb).unsqueeze(1) # [B, 1, 512]
        bge_token = self.input_norm(bge_token)
        
        # Expansion des requêtes latentes
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1) # [B, 16, 512]
        
        # Séquence de travail : [Contexte BGE | Requêtes 1..16]
        x = torch.cat([bge_token, queries], dim=1) # [B, 17, 512]
        
        # Passage dans le Transformer (Self-Attention + Context-Attention)
        x = self.transformer(x)
        
        # Extraction des 16 guides raffinés
        guides = x[:, 1:, :] # On ignore le premier token qui était le BGE
        
        encoder_outputs = BaseModelOutput(last_hidden_state=guides)
        
        if target_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Miroir sémantique (via le dernier hidden state du décodeur)
            last_hidden = outputs.decoder_hidden_states[-1].mean(dim=1)
            bge_recon = self.mirror_head(last_hidden)
            
            return outputs.loss, outputs.logits, bge_recon
        else:
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                min_length=12,
                max_length=22,
                num_beams=5,
                repetition_penalty=2.5,
                early_stopping=True
            )

if __name__ == "__main__":
    model = RosettaTransformer()
    total_params = sum(p.numel() for p in model.transformer.parameters()) / 1e6
    print(f"✅ Rosetta-V8 Attention Brain Ready: {total_params:.1f}M parameters in the Transformer.")
