import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    Provides a more effective way to encode relative positions compared to absolute embeddings.
    """
    def __init__(self, d_k, max_seq_len=512, base=10000):
        super().__init__()
        # d_k is the dimension per head (d_model // n_heads)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        # x shape: [Batch, Heads, Seq_Len, Head_Dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [L, D]

        # Returns cos and sin for the rotation formula
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rope(q, k, cos, sin):
    """
    Helper function to apply the rotation to queries and keys.
    """
    # q, k: [B, H, L, D]
    # Helper for the rotation: (x, y) -> (-y, x)
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # The actual RoPE formula: q_rotated = q*cos + rotate_half(q)*sin
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class SoftAttention_MHA(nn.Module):
    """
    Causal Multi-Head Attention using PyTorch's optimized scaled_dot_product_attention.
    Includes support for Rotary Positional Embeddings.
    """
    def __init__(self, d_model, n_heads, temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.temperature = temperature
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout_p = 0.2

    def split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, causal_mask=None, rope_cos=None, rope_sin=None):
        B, L, D = x.shape

        Q = self.split_heads(self.q_proj(x))
        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))

        # Apply rotary embeddings if provided
        if rope_cos is not None and rope_sin is not None:
            Q, K = apply_rope(Q, K, rope_cos, rope_sin)

        scale = 1.0 / math.sqrt(self.d_k)

        # Utilizing optimized flash attention if hardware supports it
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
            scale=scale
        )

        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(context)
