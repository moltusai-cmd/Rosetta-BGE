import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class RosettaSAE2T5(nn.Module):
    """
    🌀 ROSETTA-SAE-T5 (The Concept Speaker)
    - Input: SAE Latents (16384d, Sparse)
    - Converter: High-capacity Linear Projector
    - Output: T5-small Decoder
    """
    def __init__(self, sae_dim=16384, t5_name="t5-small", num_guides=8):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.d_model = self.t5.config.d_model
        self.num_guides = num_guides
        self.sae_dim = sae_dim

        # LE CONVERTER : Transforme les concepts SAE en énergie pour T5
        # 16384 -> 4096 (8 tokens * 512)
        self.converter = nn.Linear(sae_dim, num_guides * self.d_model)

    def forward(self, sae_latents, target_ids=None):
        batch_size = sae_latents.size(0)
        
        # 1. Conversion des concepts en tokens de guidage
        # On projette les activations SAE vers l'espace T5
        guides = self.converter(sae_latents).view(batch_size, self.num_guides, self.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=guides)
        
        if target_ids is not None:
            # Mode Entraînement
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=target_ids,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            # Mode Inférence
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                min_length=12,
                max_length=22,
                num_beams=4,
                repetition_penalty=2.5,
                early_stopping=True
            )

if __name__ == "__main__":
    model = RosettaSAE2T5()
    fake_sae = torch.randn(2, 16384)
    fake_targets = torch.randint(0, 32000, (2, 10))
    loss, logits = model(fake_sae, fake_targets)
    print(f"✅ Rosetta SAE-to-T5 Ready. Converter params: {16384 * 8 * 512 / 1e6:.1f} M")
