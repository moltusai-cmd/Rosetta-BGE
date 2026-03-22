import torch
import sentencepiece as spm
from sentence_transformers import SentenceTransformer, util
from model import DiffusionRosetta
import time

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Benchmarking on {device}...")

    # Load resources
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    model = DiffusionRosetta(vocab_size=sp.get_piece_size()).to(device)
    ckpt = torch.load("rosetta_v5.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_sentences = [
        "Machine learning is a field of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "A transformer is a deep learning architecture.",
        "Climate change is a global challenge for humanity.",
        "Quantum computing uses qubits for processing information.",
        "I love eating fresh apples in the morning.",
        "The capital of France is Paris.",
        "BGE embeddings are useful for retrieval tasks."
    ]

    print(f"\n{'Original Text':<50} | {'Rosetta Output':<50} | {'Sim'}")
    print("-" * 110)

    total_sim = 0
    start_time = time.time()

    for text in test_sentences:
        with torch.no_grad():
            # 1. Encode
            emb_orig = encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            
            # 2. Decode (16 steps)
            gen_ids = model.decode(emb_orig, num_steps=16)
            decoded = sp.decode(gen_ids[0].tolist())
            
            # 3. Re-encode to check semantic drift
            emb_decoded = encoder.encode([decoded], convert_to_tensor=True, normalize_embeddings=True)
            similarity = util.cos_sim(emb_orig, emb_decoded).item()
            total_sim += similarity

            print(f"{text[:48]:<50} | {decoded[:48]:<50} | {similarity:.2f}")

    avg_sim = total_sim / len(test_sentences)
    elapsed = time.time() - start_time
    
    print("-" * 110)
    print(f"✅ Benchmark Complete! Average Semantic Similarity: {avg_sim:.4f}")
    print(f"⏱️ Total time: {elapsed:.2f}s ({(elapsed/len(test_sentences)):.2f}s per sentence)")

if __name__ == "__main__":
    benchmark()
