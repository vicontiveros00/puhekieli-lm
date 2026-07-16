"""
Fine-tune a base language model on puhekieli rap + synthetic parallel pairs.

Uses LoRA for efficient fine-tuning on M4 Pro (24GB unified memory).

Usage:
    uv run python scripts/finetune.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --vocab data/tokenizer/vocab.txt \
        --train data/tokenized/rap_synthetic_train_tokens.txt \
        --valid data/tokenized/rap_synthetic_valid_tokens.txt \
        --out checkpoints/llama3.2-3b-lora-2e4-2ep \
        --epochs 2 \
        --batch 4 \
        --lora r=16
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    set_seed,
)

# Enable MPS on Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")


class PuhekieliDataset(Dataset):
    """Dataset for sequences from the tokenized text files."""

    def __init__(self, filepath: Path, tokenizer: AutoTokenizer, separator=" ", max_len=None):
        self.tokenizer = tokenizer
        self.data = []
        with open(filepath) as f:
            for line in tqdm(f, desc=f"Loading {filepath.name}"):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(separator)[:max_len] if max_len else line.split(separator)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids.append(tokenizer.eos_token_id)
                self.data.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class PuhekieliForCaptioning:
    """Seq2Seq trainer for NEU-PIEH (EN → FI) or causal LM (FI-only) fine-tuning."""

    def __init__(
        self,
        model_name: str,
        vocab_path: Path,
        train_path: Path,
        valid_path: Path,
        out_path: Path,
        max_epochs: int = 3,
        batch_size: int = 4,
        lr: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        warmup_steps: int = 250,
        eval_steps: int = 500,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(vocab_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load training data
        self.train_dataset = PuhekieliDataset(train_path, self.tokenizer)
        self.valid_dataset = PuhekieliDataset(valid_path, self.tokenizer)

        # Data load
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Setup model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.model.to(device)

        # LoRA configuration
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],  # enough for basic fine-tuning
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            print(f"PEFT/LoRA not available, training fully fine-tuned: {e}")
            # Fallback: just leave the model as is

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.eval_steps = eval_steps
        self.seed = seed

        # Training state
        self.best_valid_loss = float("inf")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            self.model.zero_grad()
            # Helper function using a simple cross-entropy objective
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1

            if self.global_step % 100 == 0:
                print(f"  Step {self.global_step}: loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{self.scheduler.T_max}: avg loss={avg_loss:.4f}")
        return avg_loss

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_best(self, valid_loss: float, epoch: int):
        """Save checkpoint if this is the best seen so far."""
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            save_path = self.out_path / "best.pt"
            self.model.save_pretrained(save_path)
            tokenizer_path = self.out_path / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            print(f"Saved new best model with valid loss={valid_loss:.4f}")

    def fit(self, max_epochs: int = 3):
        for epoch in range(max_epochs):
            if self.global_step < self.warmup_steps:
                progress = self.global_step / self.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["orig_lr"] * progress

            train_loss = self.train_epoch(self.train_loader, epoch)
            valid_loss = self.validate(self.valid_loader)
            self.save_best(valid_loss, epoch)
            self.scheduler.step()

        # Final save
        final_path = self.out_path / "final.pt"
        self.model.save_pretrained(final_path)
        print(f"Saved final model to {final_path}")
        return self.best_valid_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a base model on puhekieli")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model path or name")
    parser.add_argument("--vocab", type=Path, required=True,
                        help="Path to vocab.txt")
    parser.add_argument("--train", type=Path, required=True,
                        help="Path to tokenized train file")
    parser.add_argument("--valid", type=Path, required=True,
                        help="Path to tokenized valid file")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    for param_group in args.model.parameters():
        param_group["orig_lr"] = args.lr

    trainer = PuhekieliForCaptioning(
        model_name=args.model,
        vocab_path=args.vocab,
        train_path=args.train,
        valid_path=args.valid,
        out_path=args.out,
        max_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lora_r=args.lora_r,
        seed=args.seed,
    )

    trainer.fit(max_epochs=args.epochs)


if __name__ == "__main__":
    main()
