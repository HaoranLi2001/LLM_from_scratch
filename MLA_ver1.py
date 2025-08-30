import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class MLAconfig:
    hidden_dim: int
    num_heads: int

    q_latent_dim: int
    q_rope_dim: int
    q_nope_dim: int

    kv_latent_dim: int
    k_rope_dim: int
    k_nope_dim: int
    v_dim: int

    rope_freq: float
    max_pos_emb: int

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 较小索引位置对应较低频率
        # 较大的索引位置有较高的频率
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Multihead_Latent_Attention(nn.Module):
    def __init__(self, config):
        super().__init()

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads

        self.q_latent_dim = config.q_latent_dim
        self.q_rope_dim = config.q_rope_dim
        self.q_nope_dim = config.q_nope_dim
        self.q_dim = self.q_rope_dim + self.q_nope_dim

        self.kv_latent_dim = config.kv_latent_dim
        self.k_rope_dim = config.k_rope_dim
        self.k_nope_dim = config.k_nope_dim
        self.v_dim = config.v_dim

        self.rope_freq = config.rope_freq

        self.q_down_proj = nn.Linear(
            self.hidden_dim, 
            self.q_latent_dim,
            bias = False
        )

        self.kv_down_proj = nn.Linear(
            self.hidden_dim,
            self.kv_latent_dim,
            bias = False
        )

        self.q_up_proj = nn.Linear(
            self.q_latent_dim,
            self.q_dim,
            bias = False
        )

        self.k_up_proj = nn.Linear(
            self.kv_latent_dim,
            self.k_nope_dim,
            bias = False
        )

        self.RMSnorm = DeepseekV2RMSNorm(self.hidden_dim)
        self.RoPE = DeepseekV2RotaryEmbedding(
            self.q_rope_dim,
            self.max_pos_emb,
            self.rope_freq
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_dim,
            self.hidden_dim,
            bias = False
        )

    def forward(self, hidden_states, position_ids, attn_mask):
        bsz, seq_len, _ = hidden_states.size()

        latent_cq = self.RMSnorm(
            self.q_down_proj(hidden_states)
        )
        q = self.q_up_proj(latent_cq)
        q = q.view(
            bsz, seq_len, self.num_heads, self.q_dim
        ).transpose(1,2)
        q_nope, q_pe = torch.split(
            q, 
            [self.q_nope_dim, self.q_rope_dim], 
            dim = -1
        )

        compressed_kv = self.RMSnorm(
            self.kv_down_proj(hidden_states)
        )
        latent_ckv, k_pe = torch.split(
            compressed_kv,
            [self.kv_latent_dim, self.k_rope_dim],
            dim = -1
        )

        kv = self.k_up_proj(latent_ckv)
        kv = self.view(bsz,seq_len,self.num_heads,self.k_nope_dim+self.v_dim).transpose(1,2)

        k_nope, value = torch.split(
            kv, [self.k_nope_dim, self.v_dim],
            dim = -1
        )
        kv_seq_len = value.shape[-2]
        cos, sin = self.RoPE(value, kv_seq_len)

        q_rope, k_rope = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query = torch.cat(q_rope, q_nope)
        key = torch.cat(k_rope, k_nope)

        attn_weights = query @ key.transpose(-1,-2)
        attn_weights = torch.masked_fill(
            attn_weights,
            attn_mask == 0,
            float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim = -1)
        attn_output = attn_weights @ value
        attn_output = attn_output.transpose(1,2).view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_weights, attn_output
