import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertFeedForward(nn.Module):
    """
    Individual Expert block. 
    Uses a standard Gated Linear Unit (SiLU) architecture.
    """
    def __init__(self, d_model, d_ff, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Implementation uses SiLU (Swish) activation
        return self.fc2(self.dropout(F.silu(self.fc1(x))))

class MoE(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    Implements Top-K routing with jitter for training stability.
    """
    def __init__(self, d_model, d_ff, n_experts=6, k=2, temp=0.7):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.temperature = temp
        self.experts = nn.ModuleList([ExpertFeedForward(d_model, d_ff) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        # 1. Router with Jitter (prevents expert collapse during training)
        logits = self.router(x_flat)
        if self.training:
            # Adding small Gaussian noise to logits as a regularizer
            logits = logits + torch.randn_like(logits) * 0.01

        # Apply temperature-scaled softmax for routing probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)
        top_weights, top_indices = probs.topk(self.k, dim=-1)

        # Re-normalize weights to ensure they sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. Parallel Expert Execution
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_indices == i)
            if not mask.any(): 
                continue

            token_idx, k_idx = torch.where(mask)
            expert_out = expert(x_flat[token_idx])
            # Apply gating weights to the expert outputs
            final_output[token_idx] += expert_out * top_weights[token_idx, k_idx].unsqueeze(-1)

        # 3. Auxiliary Load Balancing Loss calculation
        tokens_per_expert = torch.bincount(top_indices.view(-1), minlength=self.n_experts).float()
        f_i = tokens_per_expert / (B * L) # Fraction of tokens per expert
        P_i = probs.mean(dim=0)           # Mean probability per expert

        # Loss is minimized when expert usage (f_i) and routing probs (P_i) are uniform
        aux_loss = self.n_experts * torch.sum(f_i * P_i)

        return final_output.view(B, L, D), aux_loss.unsqueeze(0)
