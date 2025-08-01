# deepseek_moe_gpt.py
import math, torch, inspect
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1  CONFIG
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size : int   = 1024
    vocab_size : int   = 50304   # pad-64 GPT-2 vocab
    n_layer    : int   = 12
    n_head     : int   = 12
    n_embd     : int   = 768
    dropout    : float = 0.0
    bias       : bool  = True

    # — MoE flags (DeepSeek-V3 defaults) ------------------------------------
    use_moe            : bool  = True
    num_experts        : int   = 16     # E
    top_k              : int   = 2      # tokens pick k experts
    capacity_factor    : float = 1.25   # (not enforced in vectorised impl.)
    expert_hidden_mult : int   = 4      # d_ff multiplier

# ---------------------------------------------------------------------------
# 2  UTILITY LAYERS
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
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
        self.do_attn = nn.Dropout(cfg.dropout)
        self.do_res  = nn.Dropout(cfg.dropout)
        self.flash   = hasattr(F, "scaled_dot_product_attention")
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
                dropout_p=self.do_attn.p if self.training else 0.0,
                is_causal=True
            )
        else:
            a = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
            a = a.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
            a = self.do_attn(F.softmax(a, dim=-1))
            y = a @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.do_res(self.c_proj(y))

class DenseMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.do   = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.do(self.proj(F.gelu(self.fc(x))))

# ---------------------------------------------------------------------------
# 3  VECTORISED SPARSE-MoE LAYER
# ---------------------------------------------------------------------------
class MoELayer(nn.Module):
    """
    Vectorised top-k routing (token-choice):
    * Builds a gating tensor via scatter\_
    * Runs every expert once (fused, no Python per-token loop)
      – good for research / ≤ 1 B params; use Triton/DeepSpeed later for scale
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        E, d_model = cfg.num_experts, cfg.n_embd
        d_ff       = cfg.expert_hidden_mult * d_model
        self.E, self.k = E, cfg.top_k
        self.router = nn.Linear(d_model, E, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=cfg.bias),
                nn.GELU(),
                nn.Linear(d_ff, d_model, bias=cfg.bias)
            ) for _ in range(E)
        ])
    def forward(self, x):                             # (B,L,d)
        B,L,_ = x.shape
        probs     = self.router(x).softmax(-1)        # (B,L,E)
        top_p, ix = probs.topk(self.k, dim=-1)        # (B,L,k)
        gate = torch.zeros_like(probs)                # (B,L,E)
        gate.scatter_(-1, ix, top_p)                  # fill top-k
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):     # constant 16 iterations
            y = y + gate[..., e:e+1] * expert(x)      # broadcast weight
        return x + y

# ---------------------------------------------------------------------------
# 4  TRANSFORMER BLOCK
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn= CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffn = MoELayer(cfg) if cfg.use_moe else DenseMLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn (self.ln2(x))
        return x

# ---------------------------------------------------------------------------
# 5  GPT MODEL
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg    = cfg
        self.config = cfg          # ← alias for nano-GPT scripts

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb  = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop     = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f     = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head  = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight      # weight tying

        self.apply(self._init_weights)
        for n,p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, std=0.02/math.sqrt(2*cfg.n_layer))

        print(f"Model params: {self.get_num_params()/1e6:.2f} M")

    # ---------- helpers ---------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    # ---------- forward ---------------------------------------------------
    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)

        if targets is None:                       # inference
            logits = self.lm_head(x[:,[-1],:])    # (B,1,V)
            return logits, None
        logits = self.lm_head(x)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
        return logits, loss

    # ---------- nano-GPT optimiser helper ---------------------------------
    def configure_optimizers(self, wd, lr, betas, device_type):
        decay, no_decay = [], []
        for n,p in self.named_parameters():
            if not p.requires_grad: continue
            (decay if p.dim()>=2 else no_decay).append(p)
        groups = [
            {"params":decay,    "weight_decay":wd},
            {"params":no_decay, "weight_decay":0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(
            groups, lr=lr, betas=betas,
            fused=fused_ok and device_type=="cuda"
        )

# ---------------------------------------------------------------------------
# 6  SELF-TEST & torch.compile
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg   = GPTConfig()                       # customise here if you like
    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(dev)

    x = torch.randint(0, cfg.vocab_size, (4,32), device=dev)
    with torch.no_grad():                     # eager
        y,_ = model(x)
    print("eager ok:", y.shape)

    compiled = torch.compile(model)           # torch-compile path
    with torch.no_grad():
        yc,_ = compiled(x)
    print("compiled ok:", yc.shape)
