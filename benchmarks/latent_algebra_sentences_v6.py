import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
from core.model_v6 import RosettaV6
from transformers.modeling_outputs import BaseModelOutput

def decode_vector(model, tokenizer, vector, device):
    with torch.no_grad():
        x = model.input_norm(vector)
        hidden_states = model.projector(x).view(1, 16, model.d_model)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)
        output_ids = model.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=25,
            num_beams=5,
            repetition_penalty=3.0,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def sentence_algebra():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Sentence Algebra Lab | Rosetta-V6 | Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = RosettaV6(num_guides=16).to(device)
    state_dict = torch.load('checkpoints/rosetta_v6_epoch_25.pt', map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    encoder.half()

    def get_emb(text):
        emb = encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        return emb.detach().clone().to(device).float()

    exps = [
        {
            "name": "Subject Swap",
            "op": "'A cat sitting on a mat' - 'cat' + 'dog'",
            "vec": get_emb("A cat sitting on a mat") - get_emb("cat") + get_emb("dog")
        },
        {
            "name": "Action Swap",
            "op": "'He is walking to the store' - 'walking' + 'running'",
            "vec": get_emb("He is walking to the store") - get_emb("walking") + get_emb("running")
        },
        {
            "name": "Attribute Swap",
            "op": "'The sky is very blue' - 'blue' + 'red'",
            "vec": get_emb("The sky is very blue") - get_emb("blue") + get_emb("red")
        },
        {
            "name": "Complex Scene",
            "op": "'A scientist in a lab' - 'scientist' + 'artist' - 'lab' + 'studio'",
            "vec": get_emb("A scientist in a lab") - get_emb("scientist") + get_emb("artist") - get_emb("lab") + get_emb("studio")
        }
    ]

    print("\n--- Rosetta-V6 Sentence Algebraic Results ---\n")
    for e in exps:
        vec = F.normalize(e["vec"], p=2, dim=1)
        res = decode_vector(model, tokenizer, vec, device)
        print(f"📊 {e['name']}: {e['op']}")
        print(f"🗣️  Rosetta: {res}\n" + "-"*50)

if __name__ == "__main__":
    sentence_algebra()
