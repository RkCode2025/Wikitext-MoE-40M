import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def training_step_wikitext(model, batch, optimizer, scaler, lb_weight, use_moe, scheduler, device, label_smoothing=0.1):
    """
    Performs a single training step including mixed precision and MoE auxiliary loss.
    """
    model.train()
    ids = batch["input_ids"].to(device)
    targets = batch["labels"].to(device)

    optimizer.zero_grad()

    # Mixed precision context for faster training on T4
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        out = model(ids, return_lb=use_moe)
        logits = out["logits"]

        # Causal Shift: Align logits [0...N-1] with targets [1...N]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()

        # Main Language Modeling Loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), 
            label_smoothing=label_smoothing
        )

        # MoE Load Balancing Loss
        lb_loss = out["lb_loss"].mean() if use_moe else torch.tensor(0.0, device=device)
        total_loss = loss + (lb_weight * lb_loss)

    # Gradient Scaling for float16 stability
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return loss.detach(), lb_loss.detach()

def evaluate_wikitext(model, data_loader, device):
    """
    Evaluates the model on the test set and calculates perplexity.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.inference_mode(): # Faster than no_grad
        for batch in tqdm(data_loader, desc="Evaluating"):
            ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                out = model(ids)
                logits = out["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()

                # Use reduction='sum' for mathematically exact perplexity across batches
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def plot_results(train_losses, test_losses, lb_losses):
    """
    Generates training curves for the paper/readme.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Subplot 1: Language Modeling Performance
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train LM Loss')
    if test_losses:
        plt.plot(epochs, test_losses, 'r-s', label='Test LM Loss')
    plt.title('Language Modeling Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.grid(True)

    # Subplot 2: MoE Expert Utilization
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lb_losses, 'g-^', label='MoE Aux Loss')
    plt.title('Expert Load Balancing Stability')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print(f"📊 Visualizations saved to training_metrics.png")
