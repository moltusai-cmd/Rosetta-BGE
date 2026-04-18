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

class RosettaV6(nn.Module):
    """
    🌀 ROSETTA-V6 (Master Larynx)
    - 16 Guide Tokens (1:1 Ratio)
    - Deep Residual Projector
    - Cosine Mirror Head
    """
    def __init__(self, bge_dim=384, t5_name="t5-small", num_guides=16):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.d_model = self.t5.config.d_model
        self.num_guides = num_guides

        # 1. ENTRÉE : Normalisation du BGE
        self.input_norm = nn.LayerNorm(bge_dim)

        # 2. PROJECTEUR RÉSIDUEL : 3 blocs de transformation profonde
        self.projector = nn.Sequential(
            nn.Linear(bge_dim, self.d_model),
            ResidualBlock(self.d_model),
            ResidualBlock(self.d_model),
            ResidualBlock(self.d_model),
            nn.Linear(self.d_model, num_guides * self.d_model)
        )

        # 3. MIROIR COSINE : Pour une fidélité d'angle parfaite
        self.mirror_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, bge_dim)
        )

    def forward(self, bge_emb, target_ids=None):
        batch_size = bge_emb.size(0)
        
        # Guidage 16 tokens
        x = self.input_norm(bge_emb)
        guides = self.projector(x).view(batch_size, self.num_guides, self.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=guides)
        
        if target_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extraction du sens pour le miroir
            last_hidden = outputs.decoder_hidden_states[-1].mean(dim=1)
            bge_recon = self.mirror_head(last_hidden)
            
            return outputs.loss, outputs.logits, bge_recon
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
