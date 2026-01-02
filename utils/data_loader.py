import multiprocessing
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext_loaders(tokenizer_name="gpt2", max_seq_len=512, micro_batch=32, train_subset="30%"):
    """
    Downloads and prepares WikiText datasets.
    Trains on a subset of WikiText-103 and tests on WikiText-2.
    """
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f" Loading WikiText-103 (Train: {train_subset}) and WikiText-2 (Test)...")
    
    # Load WikiText-103 for training
    train_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f'train[:{train_subset}]')
    
    # Load WikiText-2 for pure testing (Standard Benchmark)
    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')

    # 1. Tokenization Function
    def tokenize_function(examples):
        texts = [t for t in examples["text"] if len(t) > 0]
        return tokenizer(texts, truncation=False, padding=False, add_special_tokens=False)

    # 2. Grouping Function (Packing tokens into MAX_SEQ_LEN blocks)
    def group_texts(examples):
        concatenated_ids = [item for sublist in examples["input_ids"] for item in sublist]
        total_length = len(concatenated_ids)
        
        if total_length < max_seq_len:
            return {"input_ids": [], "labels": []}

        # Trim to multiple of max_seq_len
        total_length = (total_length // max_seq_len) * max_seq_len
        blocks = [concatenated_ids[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]

        return {
            "input_ids": blocks,
            "labels": blocks # Labels are identical for Causal LM (shifted in engine.py)
        }

    # Process Training Data
    tokenized_train = train_ds.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=["text"])
    lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000, num_proc=num_cpus, remove_columns=["input_ids"])

    # Process Testing Data
    tokenized_test = test_ds.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=["text"])
    lm_test = tokenized_test.map(group_texts, batched=True, batch_size=1000, num_proc=num_cpus, remove_columns=["input_ids"])

    # 3. Create DataLoaders
    train_loader = DataLoader(
        lm_train.with_format("torch"),
        batch_size=micro_batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_cpus,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        lm_test.with_format("torch"),
        batch_size=micro_batch,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    print(f"Data Loaders Ready! Train samples: {len(lm_train)} | Test samples: {len(lm_test)}")
    return train_loader, test_loader
