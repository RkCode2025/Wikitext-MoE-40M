import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

# Modular Imports
from models.transformer import Decoder_LLM
from utils.data_loader import get_wikitext_loaders
from utils.engine import training_step_wikitext, evaluate_wikitext

def parse_ablation_args():
    parser = argparse.ArgumentParser(description="MoE Ablation Training")
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--lb_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--save_path", type=str, required=True)
    # .parse_known_args() ignores extra Modal flags that cause Status 2
    args, _ = parser.parse_known_args()
    return args

args = parse_ablation_args()

class Config:
    D_MODEL, D_FF, N_HEADS, N_LAYERS = 512, 1536, 8, 8
    N_EXPERTS = args.num_experts
    K_MOE = args.top_k
    LB_LOSS_WEIGHT = args.lb_weight
    NUM_EPOCHS = args.epochs
    SAVE_PATH = args.save_path
    MAX_SEQ_LEN, MICRO_BATCH = 512, 32
    LEARNING_RATE, WEIGHT_DECAY = 5e-4, 0.1
    WARMUP_EPOCHS = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = Config()
    train_loader, test_loader = get_wikitext_loaders(cfg.MAX_SEQ_LEN, cfg.MICRO_BATCH)

    model = Decoder_LLM(
        vocab_size=50257, d_model=cfg.D_MODEL, max_seq_len=cfg.MAX_SEQ_LEN,
        n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF, n_heads=cfg.N_HEADS,
        n_experts=cfg.N_EXPERTS, k_moe=cfg.K_MOE
    ).to(cfg.DEVICE)

    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=cfg.LEARNING_RATE, 
                           total_steps=len(train_loader) * cfg.NUM_EPOCHS, pct_start=0.1)

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        use_moe = epoch >= cfg.WARMUP_EPOCHS
        for batch in train_loader:
            training_step_wikitext(model, batch, optimizer, scaler, 
                                   cfg.LB_LOSS_WEIGHT, use_moe, None, cfg.DEVICE)
            scheduler.step()

        # Validation (This print is captured by final_work.py)
        _, avg_ppl = evaluate_wikitext(model, test_loader, cfg.DEVICE)
        print(f"Epoch {epoch+1} | Test PPL: {avg_ppl:.2f}")

    torch.save(model.state_dict(), cfg.SAVE_PATH)
    print(f"Ablation Complete. Saved to {cfg.SAVE_PATH}")

if __name__ == "__main__":
    main()
