import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SoftAttention_MHA, RoPE
from .moe import MoE

class Transformer_Block(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_experts, k_moe, temp):
        super().__init__()
        # RMSNorm is excellent for stability in MoE architectures
        self.ln1 = nn.RMSNorm(d_model, eps=1e-6) 
        self.attn = SoftAttention_MHA(d_model, n_heads, temperature=temp)
        self.ln2 = nn.RMSNorm(d_model, eps=1e-6)
        self.moe = MoE(d_model, d_ff, n_experts, k_moe, temp)

        self.attn_dropout = nn.Dropout(0.3)
        self.moe_dropout = nn.Dropout(0.3)

    def forward(self, x, mask, cos, sin):
        # 1. Attention path with Residual Connection
        norm_x = self.ln1(x)
        x = x + self.attn_dropout(self.attn(norm_x, mask, cos, sin))

        # 2. MoE path with Residual Connection
        norm_x2 = self.ln2(x)
        moe_out, lb_loss = self.moe(norm_x2)
        x = x + self.moe_dropout(moe_out)

        return x, lb_loss

class Decoder_LLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, n_layers, d_ff, n_heads, n_experts, k_moe, temp):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # RoPE provides better long-range context than absolute embeddings
        self.rope = RoPE(d_model // n_heads, max_seq_len)

        self.layers = nn.ModuleList([
            Transformer_Block(d_model, d_ff, n_heads, n_experts, k_moe, temp)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.RMSNorm(d_model, eps=1e-6)

        # Weight Tying: Reduces parameter count and improves performance
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids, return_lb=False):
        B, L = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        cos, sin = self.rope(x, L)

        # Causal mask ensures tokens only attend to the past
        mask = torch.tril(torch.ones(L, L, device=device)).bool()

        total_lb = 0.0
        for layer in self.layers:
            x, lb = layer(x, mask, cos, sin)
            total_lb += lb

        logits = self.head(self.ln_f(x))
        return {
            "logits": logits,
            "lb_loss": total_lb if return_lb else torch.zeros(1, device=device)
        }
