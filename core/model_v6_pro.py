import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class RosettaV6Pro(nn.Module):
    """
    🌀 ROSETTA-V6 PRO (Giga-Boost Edition)
    - 16 Guide Tokens
    - 10 Deep Residual Blocks (High Capacity)
    - 1024 Hidden Dimensions
    - Linear Expansion from BGE 384
    """
    def __init__(self, bge_dim=384, t5_name="t5-small", num_guides=16):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.d_model = 1024
        self.t5_dim = self.t5.config.d_model # 512
        self.num_guides = num_guides

        # 1. ENTRÉE : Expansion sémantique
        self.input_expander = nn.Linear(bge_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

        # 2. LE CERVEAU PRO : 10 Blocs de réflexion profonde
        self.projector = nn.Sequential(*[ResidualBlock(self.d_model) for _ in range(10)])

        # 3. SORTIE : Projection vers T5
        self.output_projector = nn.Linear(self.d_model, num_guides * self.t5_dim)

        # 4. MIROIR (Toujours présent pour la structure)
        self.mirror_head = nn.Sequential(
            nn.Linear(self.t5_dim, self.t5_dim),
            nn.GELU(),
            nn.Linear(self.t5_dim, bge_dim)
        )

    def get_guides(self, bge_emb):
        """Utilaire pour extraire uniquement les guides (utilisé par la Forge)"""
        x = self.input_expander(bge_emb)
        x = self.input_norm(x)
        x = self.projector(x)
        guides = self.output_projector(x).view(bge_emb.size(0), self.num_guides, self.t5_dim)
        return guides

    def forward(self, bge_emb, target_ids=None):
        batch_size = bge_emb.size(0)
        
        # Phase de projection PRO via l'utilitaire
        guides = self.get_guides(bge_emb)
        encoder_outputs = BaseModelOutput(last_hidden_state=guides)
        
        if target_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                return_dict=True
            )
            return outputs.loss, outputs.logits, None # Pas de miroir en train direct
        else:
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                min_length=8,
                max_length=48,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

if __name__ == "__main__":
    model = RosettaV6Pro()
    # Calcul des paramètres du projecteur (sans T5)
    proj_params = (sum(p.numel() for p in model.projector.parameters()) + 
                   sum(p.numel() for p in model.input_expander.parameters()) +
                   sum(p.numel() for p in model.output_projector.parameters()))
    print(f"✅ V6 PRO Projector Ready: {proj_params/1e6:.1f}M parameters.")
