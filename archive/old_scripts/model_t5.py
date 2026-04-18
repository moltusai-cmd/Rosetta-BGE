import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

class RosettaT5(nn.Module):
    """
    🌀 ROSETTA-T5 PRO (Surgical Edition)
    - 8 Guide Tokens (High-Res)
    - Semantic Mirror Head (Fidelity)
    """
    def __init__(self, model_name="t5-small", bge_dim=384, num_guides=8):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.d_model = self.t5.config.d_model 
        self.num_guides = num_guides
        
        # Projecteur Haute-Résolution (8 tokens)
        self.bge_projector = nn.Sequential(
            nn.Linear(bge_dim, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model * num_guides) 
        )

        # Tête Miroir : Reconstruit le BGE depuis la sortie du décodeur
        self.mirror_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, bge_dim)
        )

    def forward(self, bge_emb, target_ids=None):
        batch_size = bge_emb.size(0)
        
        # 1. Projeter en 8 tokens de guidage
        hidden_states = self.bge_projector(bge_emb).view(batch_size, self.num_guides, self.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)
        
        if target_ids is not None:
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Reconstruction du BGE à partir de la moyenne des états du décodeur
            # Cela force T5 à maintenir le sens pur dans ses couches internes
            last_hidden = outputs.decoder_hidden_states[-1].mean(dim=1)
            bge_recon = self.mirror_head(last_hidden)
            
            return outputs.loss, outputs.logits, bge_recon
        else:
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                min_length=12,
                max_length=22,
                num_beams=4,
                repetition_penalty=2.5,
                early_stopping=True
            )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RosettaT5().to(device)
    fake_bge = torch.randn(2, 384).to(device)
    fake_targets = torch.randint(0, 32000, (2, 10)).to(device)
    loss, logits, recon = model(fake_bge, fake_targets)
    print(f"✅ Rosetta-T5 PRO Ready. Recon shape: {recon.shape}")
