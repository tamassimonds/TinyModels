"""
Minimal GPT-2-style LM + optional DeepSeek-style Sparse-MoE.
Torch-compile-friendly (no Python-side loops over dynamic data).
"""

import math, inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50_304          # padded GPT-2 vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True                 # biases in Linear / LN
    # --- MoE flags ---------------------------------------------------------
    use_moe: bool = True
    num_experts: int = 16
    top_k: int = 2
    capacity_factor: float = 1.25     # (not enforced in this vectorised impl)
    expert_hidden_mult: int = 4       # FF dim multiplier

# ---------------------------------------------------------------------------
# 2. BUILDING BLOCKS
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch 2.x still lacks bias=False)."""
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                      .view(1,1,cfg.block_size,cfg.block_size)
            )

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(C, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,k,v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y   = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.c_proj(y))

class DenseMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.do   = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.do(self.proj(F.gelu(self.fc(x))))

# ---------------------------------------------------------------------------
# 3. VECTORISED MoE LAYER  (torch.compile-friendly)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    Vectorised top-k router:
    • computes full softmax
    • keeps top-k weights per token via scatter (no Python lists)
    • runs *every* expert, but weights are zero for inactive tokens
      – simple & fully fuse-able; good for toy-scale or research
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.E      = cfg.num_experts
        self.k      = cfg.top_k
        d_model     = cfg.n_embd
        d_ff        = cfg.expert_hidden_mult * d_model

        self.router = nn.Linear(d_model, self.E, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=cfg.bias),
                nn.GELU(),
                nn.Linear(d_ff, d_model, bias=cfg.bias)
            ) for _ in range(self.E)
        ])

    def forward(self, x):                       # x: (B, L, d_model)
        B, L, _ = x.shape
        probs    = self.router(x).softmax(dim=-1)        # (B,L,E)
        top_p, top_idx = probs.topk(self.k, dim=-1)      # (B,L,k)

        # Build gating tensor with zeros elsewhere (scatter_ is all-GPU)
        gating = torch.zeros_like(probs)                 # (B,L,E)
        gating.scatter_(-1, top_idx, top_p)

        # Expert outputs & gated sum
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):        # loop over constant E
            y = y + gating[..., e:e+1] * expert(x)       # broadcast gate

        return x + y                                     # residual

# ---------------------------------------------------------------------------
# 4. TRANSFORMER BLOCK
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffn = MoELayer(cfg) if cfg.use_moe else DenseMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn (self.ln2(x))
        return x

# ---------------------------------------------------------------------------
# 5. GPT MODEL
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying

        # init
        self.apply(self._init_weights)
        for n,p in self.named_parameters():
            if n.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*cfg.n_layer))
        total = sum(p.numel() for p in self.parameters())/1e6
        print(f"Model params: {total:.2f} M")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)

        if targets is None:                         # inference
            logits = self.lm_head(x[:,[-1],:])      # last token only
            return logits, None
        logits = self.lm_head(x)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
        return logits, loss

# ---------------------------------------------------------------------------
# 6. QUICK TEST + COMPILE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg).to("cuda" if torch.cuda.is_available() else "cpu")

    # dummy batch
    B, T = 4, 32
    x = torch.randint(0, cfg.vocab_size, (B,T), device=next(model.parameters()).device)

    # eager
    with torch.no_grad():
        y,_ = model(x)
    print("eager OK :", y.shape)

    # compiled
    compiled = torch.compile(model)
    with torch.no_grad():
        yc,_ = compiled(x)
    print("compiled OK:", yc.shape)
