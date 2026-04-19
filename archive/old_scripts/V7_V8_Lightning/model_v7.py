import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class GigaBlock(nn.Module):
    """Bloc Résiduel avec Pre-Norm pour la stabilité profonde"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # Pre-Norm : le gradient circule plus facilement dans le 'x +'
        return x + self.net(self.norm(x))

class RosettaV7(nn.Module):
    def __init__(self, bge_dim=384, t5_name="t5-small", num_guides=16):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.d_model = self.t5.config.d_model # 512
        self.num_guides = num_guides

        # 1. ENTRÉE : Projection vers l'espace de calcul (2048d)
        self.input_expander = nn.Linear(bge_dim, 2048)
        self.input_norm = nn.LayerNorm(2048)
        
        # 2. LE CERVEAU : 12 couches de réflexion profonde (Pre-Norm)
        self.brain = nn.Sequential(*[GigaBlock(2048) for _ in range(12)])

        # 3. SORTIE : Projection vers les 16 guides T5
        self.output_projector = nn.Linear(2048, num_guides * self.d_model)
        
        # 4. MIROIR
        self.mirror_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, bge_dim)
        )

    def forward(self, bge_emb, target_ids=None):
        batch_size = bge_emb.size(0)
        
        x = self.input_expander(bge_emb)
        x = self.input_norm(x)
        x = self.brain(x)
        
        guides = self.output_projector(x).view(batch_size, self.num_guides, self.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=guides)
        
        if target_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
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
                no_repeat_ngram_size=3,
                early_stopping=True
            )
