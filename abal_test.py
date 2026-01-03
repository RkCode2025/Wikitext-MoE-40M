import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import multiprocessing
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 1. Modular Imports 
from models.transformer import Decoder_LLM

def parse_test_args():
    parser = argparse.ArgumentParser(description="MoE Ablation Benchmarking")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    # Using parse_known_args to stay consistent with the train script
    args, _ = parser.parse_known_args()
    return args

def get_test_only_loader(tokenizer, max_seq_len=512, batch_size=32):
    """Standardizes the WikiText-2 Test set for benchmarking."""
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
    """Benchmark Engine: Calculates Perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(ids)
                # Handle cases where model returns a dict or a raw tensor
                logits = out["logits"] if isinstance(out, dict) else out

                # Causal Shift
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1),
                    reduction='sum'
                )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return avg_loss, perplexity

def main():
    args = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Build Model matching the ablation architecture
    model = Decoder_LLM(
        vocab_size=50257,
        d_model=512,
        max_seq_len=512,
        n_layers=8,
        d_ff=1536,
        n_heads=8,
        n_experts=args.num_experts,
        k_moe=args.top_k
    ).to(device)

    # 3. Load weights from the volume path
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Successfully loaded {args.model_path}")
    else:
        print(f"❌ Model path not found: {args.model_path}")
        return

    # 4. Benchmark
    test_loader = get_test_only_loader(tokenizer)
    run_final_performance_test(model, test_loader, device)

if __name__ == "__main__":
    main()
