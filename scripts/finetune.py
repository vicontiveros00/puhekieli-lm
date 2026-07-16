"""
Fine-tune a base language model on puhekieli rap + synthetic parallel pairs.

Uses HuggingFace tokenizers (no custom vocab required).

Usage:
    uv run python scripts/finetune.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --fi-data data/clean/genius_rap.jsonl \
        --fi-en data/clean/rap_synthetic.jsonl \
        --out checkpoints/llama3.2-3b-lora-2e-2ep
"""

import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    set_seed,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")


class PuhekieliFiDataset(Dataset):
    """Dataset for FI-only text (genius_rap)."""

    def __init__(self, file_path: Path, tokenizer: AutoTokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path) as f:
            for line in tqdm(f, desc=f"Loading {file_path.name}"):
                line = line.strip()
                if not line:
                    continue
                tokens = tokenizer(
                    line,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors=None,
                )
                tokens = [int(t) for t in tokens["input_ids"]]
                tokens.append(tokenizer.eos_token_id)
                if len(tokens) > max_len + 2:
                    tokens = tokens[:max_len + 2]
                self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PuhekieliTrainSet(Dataset):
    """Dataset for parallel FI-FI training pairs (synthetic back-translation)."""

    def __init__(self, file_path: Path, tokenizer: AutoTokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path) as f:
            for line in tqdm(f, desc=f"Loading {file_path.name}", total=sum(1 for _ in f)):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Should have {fi: "..."} in synthetic
                fi_text = obj.get("fi", obj.get("text", ""))
                tokens = tokenizer(
                    fi_text,
                    truncation=True,
                    max_length=max_len,
                    return_tensors=None,
                )
                tokens = [int(t) for t in tokens["input_ids"]]
                tokens.append(tokenizer.eos_token_id)
                if len(tokens) > max_len + 2:
                    tokens = tokens[:max_len + 2]
                self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PuhekieliFineTuner:
    """Fine-tune puhekieli model (FI-only estimate FI + EN→FI)."""

    def __init__(
        self,
        model_name: str,
        gensfi_path: Path,
        train_path: Path,
        valid_path: Path,
        out_path: Path,
        max_epochs: int = 2,
        batch_size: int = 4,
        lr: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        warmup_steps: int = 250,
        eval_steps: int = 500,
        max_len: int = 512,
        seed: int = 42,
    ):
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len

        # Load datasets
        self.train_dataset = PuhekieliTrainSet(train_path, self.tokenizer, max_len=max_len)
        self.valid_dataset = PuhekieliTrainSet(valid_path, self.tokenizer, max_len=max_len)
        print(f"Loaded {len(self.train_dataset)} train / {len(self.valid_dataset)} valid samples")

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.model.to(device)

        # LoRA
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            print(f"LoRA not available, training fully fine-tuned: {e}")

        # Training
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.eval_steps = eval_steps
        self.seed = seed
        self.best_valid_loss = float("inf")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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

    def save_checkpoint(self, valid_loss: float, epoch: int):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            save_path = self.out_path / "best.pt"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path / "tokenizer")
            print(f"Saved new best: valid_loss={valid_loss:.4f}")

    def fit(self, max_epochs: int = 2):
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(self.train_loader, epoch)
            valid_loss = self.validate(self.valid_loader)
            self.save_checkpoint(valid_loss, epoch)
            self.scheduler.step()

        # Final save
        final_path = self.out_path / "final.pt"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path / "tokenizer")
        print(f"Saved final to {final_path}")
        return self.best_valid_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune puhekieli model")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--fi-data", type=Path, required=True)
    parser.add_argument("--fi-en", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    trainer = PuhekieliFineTuner(
        model_name=args.model,
        gensfi_path=args.fi_data,
        train_path=args.fi_en,
        valid_path=args.fi_en,
        out_path=args.out,
        max_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lora_r=args.lora_r,
        max_len=args.max_len,
        seed=args.seed,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
