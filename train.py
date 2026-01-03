import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

# 1. Modular Imports
from models.transformer import Decoder_LLM
from utils.data_loader import get_wikitext_loaders
from utils.engine import training_step_wikitext, evaluate_wikitext, plot_results

# 2. Research Configuration
class Config:
    D_MODEL = 512
    D_FF = 1536
    N_HEADS = 8
    N_LAYERS = 8
    N_EXPERTS = 6
    K_MOE = 2
    MAX_SEQ_LEN = 512
    MICRO_BATCH = 32
    ACCUMULATION_STEPS = 4  # Global Batch Size = 32 * 4 = 128
    NUM_EPOCHS = 4
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.1
    LB_LOSS_WEIGHT = 0.1
    WARMUP_EPOCHS = 1       # Set to 1 if you want to delay MoE penalty
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = Config()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 3. Data Loading
    train_loader, test_loader = get_wikitext_loaders(
        max_seq_len=cfg.MAX_SEQ_LEN, 
        micro_batch=cfg.MICRO_BATCH
    )

    # 4. Model Setup
    model = Decoder_LLM(
        vocab_size=50257, # GPT-2 Tokenizer standard
        d_model=cfg.D_MODEL,
        max_seq_len=cfg.MAX_SEQ_LEN,
        n_layers=cfg.N_LAYERS,
        d_ff=cfg.D_FF,
        n_heads=cfg.N_HEADS,
        n_experts=cfg.N_EXPERTS,
        k_moe=cfg.K_MOE,
        temp=0.7
    ).to(cfg.DEVICE)

    # Compile for 20% speedup on T4
    model = torch.compile(model)

    # Initializing weights using GPT-2 standard (std=0.02)
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.RMSNorm):
            nn.init.ones_(m.weight)
    model.apply(init_weights)

    # 5. Optimization
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler()
    
    total_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=cfg.LEARNING_RATE, 
        total_steps=total_steps, 
        pct_start=0.1
    )

    # 6. Training Loop
    history = {"train_loss": [], "test_loss": [], "lb_loss": []}
    
    for epoch in range(cfg.NUM_EPOCHS):
        use_moe = epoch >= cfg.WARMUP_EPOCHS
        epoch_lm_loss, epoch_lb_loss = 0, 0

        for i, batch in enumerate(train_loader):
            # Training Step
            lm_loss, lb_loss = training_step_wikitext(
                model, batch, optimizer, scaler, 
                cfg.LB_LOSS_WEIGHT, use_moe, None, cfg.DEVICE
            )
            
            epoch_lm_loss += lm_loss.item()
            epoch_lb_loss += lb_loss.item()

            # Gradient Accumulation & Clipping
            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        # Validation
        avg_test_loss, avg_ppl = evaluate_wikitext(model, test_loader, cfg.DEVICE)
        
        history["train_loss"].append(epoch_lm_loss / len(train_loader))
        history["test_loss"].append(avg_test_loss)
        history["lb_loss"].append(epoch_lb_loss / len(train_loader))

        print(f"Epoch {epoch+1} | Test PPL: {avg_ppl:.2f}")

    # 7. Finalize
    plot_results(history["train_loss"], history["test_loss"], history["lb_loss"])
    torch.save(model.state_dict(), "raptor_moe_final.pt")
    print("Training Complete. Model saved as raptor_moe_final.pt")

if __name__ == "__main__":
    main()
