import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

class SwiGLU(nn.Module):
    """ Modern Gated MLP activation used in Llama/Mistral """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RosettaBlock(nn.Module):
    """ A single Transformer block optimized for Rosetta """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_model * 4)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, cross_kv=None):
        # Self-Attention or Cross-Attention if cross_kv is provided
        # For simplicity in this v6, we stay on Self-Attention but with Guide Tokens
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

class DiffusionRosettaV6(nn.Module):
    """
    🌀 ROSETTA ULTRA v6.0
    - SwiGLU MLP
    - Weight Tying
    - Enhanced Semantic Guidance
    """
    def __init__(self, vocab_size, bge_dim=384, d_model=1024, n_heads=16, num_cycles=6, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.num_cycles = num_cycles

        # 1. Conditioning Projector
        self.bge_projector = nn.Sequential(
            nn.Linear(bge_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4) # 4 Guide tokens
        )

        # 2. Text Embeddings
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        
        # 3. Position Embeddings (Absolute for now, RoPE integration planned)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens + 4, d_model))

        # 4. Optimized Recursive Layers
        self.block_1 = RosettaBlock(d_model, n_heads)
        self.block_2 = RosettaBlock(d_model, n_heads)

        # 5. Output Head
        self.norm_f = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size + 1, bias=False)
        
        # WEIGHT TYING: Share weights between embedding and output
        self.fc_out.weight = self.token_emb.weight 

        # 6. Semantic Mirror
        self.semantic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, bge_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, bge_emb, current_tokens, return_semantic=False):
        batch_size = bge_emb.size(0)
        
        # Guide Tokens (The Nose)
        guides = self.bge_projector(bge_emb).view(batch_size, 4, self.d_model)
        
        # Token Embeddings
        t_emb = self.token_emb(current_tokens)
        
        # Concatenate and add positions
        x = torch.cat([guides, t_emb], dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        
        # --- RECURSIVE REFINEMENT ---
        for _ in range(self.num_cycles):
            x = self.block_1(x)
            x = self.block_2(x)
        # ---------------------------
        
        # Extract only the text tokens (not guides)
        features = self.norm_f(x[:, 4:, :])
        logits = self.fc_out(features)
        
        if return_semantic:
            semantic_pred = self.semantic_head(features.mean(dim=1))
            return logits, semantic_pred
            
        return logits

    @torch.no_grad()
    def decode(self, bge_emb, num_steps=12):
        self.eval()
        batch_size = bge_emb.size(0)
        device = bge_emb.device
        tokens = torch.full((batch_size, self.num_tokens), self.mask_id, dtype=torch.long, device=device)
        
        for t in range(num_steps):
            logits = self.forward(bge_emb, tokens)
            probs = F.softmax(logits, dim=-1)
            confidence, predictions = torch.max(probs, dim=-1)
            
            num_to_mask = max(0, int(self.num_tokens * (1 - (t + 1) / num_steps)))
            if num_to_mask > 0:
                _, mask_indices = torch.topk(confidence, num_to_mask, largest=False, dim=-1)
                tokens = predictions
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                tokens[batch_indices, mask_indices] = self.mask_id
            else:
                tokens = predictions
        return tokens
