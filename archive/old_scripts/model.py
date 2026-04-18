import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

class DiffusionRosetta(nn.Module):
    """
    🌀 ROSETTA PRO v5.2 - "The 70M Brain"
    Dual-layer recursion with d_model=1024.
    """
    def __init__(self, vocab_size, bge_dim=384, d_model=1024, n_heads=16, num_cycles=6, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.num_cycles = num_cycles

        # 1. Conditioning (The Nose)
        self.bge_projector = nn.Sequential(
            nn.Linear(bge_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4) # 4 guide tokens
        )

        # 2. Text Embeddings
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens + 4, d_model))

        # 3. DUAL RECURSIVE Transformer Layers
        self.layer_1 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.layer_2 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )

        self.norm_f = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # 4. Semantic Mirror
        self.semantic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, bge_dim)
        )

    def forward(self, bge_emb, current_tokens, return_semantic=False):
        batch_size = bge_emb.size(0)
        guides = self.bge_projector(bge_emb).view(batch_size, 4, self.d_model)
        t_emb = self.token_emb(current_tokens)
        
        x = torch.cat([guides, t_emb], dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        
        # --- DUAL RECURSIVE LOOP ---
        for _ in range(self.num_cycles):
            x = self.layer_1(x)
            x = self.layer_2(x)
        # ---------------------------
        
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
