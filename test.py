import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import multiprocessing
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_test_only_loader(tokenizer, max_seq_len=512, batch_size=32):
    """
    Standardizes the WikiText-2 Test set for benchmarking.
    Ensures zero padding and full block alignment.
    """
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def tokenize_map(examples):
        return tokenizer(examples["text"], truncation=False, padding=False, add_special_tokens=False)

    tokenized_ds = dataset.map(tokenize_map, batched=True, num_proc=num_cpus, remove_columns=["text"])

    def group_map(examples):
        concatenated_ids = [item for sublist in examples["input_ids"] for item in sublist]
        total_length = (len(concatenated_ids) // max_seq_len) * max_seq_len
        blocks = [concatenated_ids[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
        
        return {
            "input_ids": blocks, 
            "labels": blocks,
            "attention_mask": [[1] * max_seq_len for _ in blocks]
        }

    test_dataset = tokenized_ds.map(
        group_map, 
        batched=True, 
        num_proc=num_cpus, 
        remove_columns=tokenized_ds.column_names,
        desc="Aligning WikiText-2 Test Blocks"
    )
    
    return DataLoader(
        test_dataset.with_format("torch"), 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )

def run_final_performance_test(model, test_loader, device):
    """
    The Benchmark Engine. Calculates Perplexity based on total token cross-entropy.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print(f" Benchmarking initiated on {device}...")
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                out = model(ids)
                logits = out["logits"]

                # Causal Shift: Aligning prediction [t] with ground truth [t+1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()

                # 'sum' is used here to avoid averaging twice, ensuring precision
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1),
                    reduction='sum'
                )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"   WIKITEXT-2 TEST BENCHMARK")
    print("   " + "-" * 37)
    print(f"   Avg Loss:   {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print("   " + "="*37)
    
    return avg_loss, perplexity
