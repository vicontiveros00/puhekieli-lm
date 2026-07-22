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
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, # Changed from AutoModelForSeq2SeqLM
    set_seed,
    DataCollatorForLanguageModeling, # Added this
    get_cosine_schedule_with_warmup,
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
        self.example_text = None  # first formatted prompt, for a sanity peek
        with open(file_path, 'r') as f:
            raw_lines = f.readlines() # Read all lines once

        for line in tqdm(raw_lines, desc=f"Loading {file_path.name}"):
            line = line.strip()
            if not line:
                continue

            # For FI-only data, the input can be a generic prompt to generate Finnish
            # The labels will be the tokenized Finnish text following the prompt.
            messages = [
                {"role": "user", "content": "Generate Finnish text:"},
                {"role": "assistant", "content": line},
            ]

            if self.example_text is None:
                self.example_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

            tokenized_output = tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                padding="max_length", # Ensure all sequences are padded to max_len
                add_generation_prompt=False,
            )
            
            # Extract 1D tensors from BatchEncoding
            self.data.append({
                "input_ids": tokenized_output["input_ids"].squeeze(0),
                "attention_mask": tokenized_output["attention_mask"].squeeze(0),
                "labels": tokenized_output["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PuhekieliTrainSet(Dataset):
    """Dataset for parallel EN->FI training pairs (synthetic back-translation)."""

    def __init__(self, file_path: Path, tokenizer: AutoTokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.data = []
        self.example_text = None  # first formatted prompt, for a sanity peek
        with open(file_path, 'r') as f:
            raw_lines = f.readlines() # Read all lines once

        for line in tqdm(raw_lines, desc=f"Loading {file_path.name}"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            en_text = obj.get("en", "")
            fi_text = obj.get("fi", "")

            messages = [
                {"role": "user", "content": f"Translate to Finnish: {en_text}"},
                {"role": "assistant", "content": fi_text},
            ]

            if self.example_text is None:
                self.example_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            
            tokenized_output = tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                padding="max_length", # Ensure all sequences are padded to max_len
                add_generation_prompt=False, # No generation prompt here, as the assistant message is the target
            )

            # For Causal LMs, labels are typically a copy of input_ids, with padding handled by collator
            # Extract 1D tensors from BatchEncoding
            self.data.append({
                "input_ids": tokenized_output["input_ids"].squeeze(0),
                "attention_mask": tokenized_output["attention_mask"].squeeze(0),
                "labels": tokenized_output["input_ids"].squeeze(0),
            })

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
        log_every: int = 20,
        dry_run: bool = False,
    ):
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.dry_run = dry_run
        self.max_epochs = max_epochs
        self.global_step = 0
        self.best_valid_loss = float("inf")

        print("=" * 64)
        print("puhekieli fine-tune")
        print("=" * 64)
        print(f"  model        : {model_name}")
        print(f"  device       : {device}  (dtype: bfloat16)")
        print(f"  fi-only data : {gensfi_path}")
        print(f"  parallel data: {train_path}")
        print(f"  out dir      : {self.out_path}")
        print(f"  epochs       : {max_epochs}   batch: {batch_size}   max-len: {max_len}")
        print(f"  lr           : {lr}   warmup-steps: {warmup_steps}")
        print(f"  lora         : r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
        print(f"  seed         : {seed}   log-every: {log_every} steps"
              + ("   [DRY RUN]" if dry_run else ""))
        print("=" * 64)

        # Load tokenizer
        print("\n[1/4] Loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len

        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Load datasets
        print("\n[2/4] Loading datasets ...")
        # Split rap_synthetic.jsonl into train/valid
        full_synthetic_dataset = PuhekieliTrainSet(train_path, self.tokenizer, max_len=max_len)
        train_size = int(0.9 * len(full_synthetic_dataset))
        valid_size = len(full_synthetic_dataset) - train_size
        synthetic_train_dataset, synthetic_valid_dataset = torch.utils.data.random_split(
            full_synthetic_dataset, [train_size, valid_size]
        )

        # Load FI-only data (genius_rap.jsonl)
        fi_only_dataset = PuhekieliFiDataset(gensfi_path, self.tokenizer, max_len=max_len)

        # Combine FI-only with synthetic training data
        self.train_dataset = torch.utils.data.ConcatDataset(
            [fi_only_dataset, synthetic_train_dataset]
        )
        self.valid_dataset = synthetic_valid_dataset # Validation only from parallel data

        # Data-composition summary
        n_fi = len(fi_only_dataset)
        n_syn_train = len(synthetic_train_dataset)
        n_valid = len(self.valid_dataset)
        print("\n  data composition:")
        print(f"    FI-only (flavor)    : {n_fi:>6}")
        print(f"    synthetic EN->FI    : {n_syn_train:>6}  (train)  {n_valid:>6}  (valid)")
        print(f"    ----------------------------------------")
        print(f"    total train         : {len(self.train_dataset):>6}")
        print(f"    total valid         : {n_valid:>6}  (parallel only)")

        # One decoded example so the chat-template formatting is visible
        example = getattr(full_synthetic_dataset, "example_text", None) \
            or getattr(fi_only_dataset, "example_text", None)
        if example:
            print("\n  example formatted prompt (parallel):")
            for ln in example.rstrip().splitlines():
                print(f"    | {ln}")

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.data_collator,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * max_epochs

        if self.dry_run:
            print("\n[dry-run] datasets + config verified; skipping model load and training.")
            print(f"[dry-run] would run {self.total_steps} optimizer steps "
                  f"({self.steps_per_epoch}/epoch x {max_epochs} epochs).")
            self.model = None
            return

        # Model
        print("\n[3/4] Loading model (this can take a moment) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # Qwen3-4B is in BF16
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
        except ImportError:
            print("\n" + "!" * 64)
            print("!! WARNING: peft not installed -> falling back to FULL fine-tuning.")
            print("!! A multi-billion-param model in full FT will very likely OOM on MPS.")
            print("!! Install with:  uv sync --extra finetune")
            print("!" * 64 + "\n")

        # Training
        print("\n[4/4] Setting up optimizer + scheduler ...")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # Proper per-step cosine schedule WITH warmup (was previously a per-epoch
        # no-op: warmup_steps was stored but never applied).
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(warmup_steps, max(1, self.total_steps // 10)),
            num_training_steps=self.total_steps,
        )
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.seed = seed
        print(f"      scheduler: cosine, warmup={min(warmup_steps, max(1, self.total_steps // 10))} "
              f"/ {self.total_steps} total steps")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        running = 0.0  # running-avg loss for a smoother signal than a single batch
        t0 = time.time()
        pbar = tqdm(
            loader,
            desc=f"epoch {epoch+1}/{self.max_epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()      # per-step schedule (warmup + cosine)
            self.optimizer.zero_grad()
            self.global_step += 1

            batch_loss = loss.item()
            total_loss += batch_loss
            running = batch_loss if i == 0 else 0.9 * running + 0.1 * batch_loss
            lr = self.scheduler.get_last_lr()[0]

            # live in-bar readout every step; periodic printed line for logs
            pbar.set_postfix(loss=f"{running:.4f}", lr=f"{lr:.2e}")
            if self.global_step % self.log_every == 0:
                tqdm.write(
                    f"  step {self.global_step}/{self.total_steps}  "
                    f"loss={running:.4f}  lr={lr:.2e}"
                )

        avg_loss = total_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{self.max_epochs}: avg loss={avg_loss:.4f}  "
              f"({dt/60:.1f} min, {len(loader)/max(dt,1e-9):.2f} it/s)")
        return avg_loss

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="validating", leave=False, dynamic_ncols=True):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, valid_loss: float, epoch: int):
        if valid_loss < self.best_valid_loss:
            prev = self.best_valid_loss
            self.best_valid_loss = valid_loss
            save_path = self.out_path / "best.pt"
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path / "tokenizer")
            delta = "" if prev == float("inf") else f" (was {prev:.4f})"
            print(f"  ✓ new best valid_loss={valid_loss:.4f}{delta} -> saved to {save_path}")
        else:
            print(f"  · no improvement (valid_loss={valid_loss:.4f}, "
                  f"best={self.best_valid_loss:.4f}) -> not saving")

    def fit(self, max_epochs: int = None):
        max_epochs = max_epochs or self.max_epochs
        if self.dry_run:
            print("\n[dry-run] nothing to train. exiting.")
            return self.best_valid_loss
        print("\n" + "=" * 64)
        print(f"Training: {max_epochs} epoch(s), {self.total_steps} total steps")
        print("=" * 64)
        t_start = time.time()
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(self.train_loader, epoch)
            valid_loss = self.validate(self.valid_loader)
            self.save_checkpoint(valid_loss, epoch)

        # Final save
        final_path = self.out_path / "final.pt"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path / "tokenizer")
        total_min = (time.time() - t_start) / 60
        print("\n" + "=" * 64)
        print("Done.")
        print(f"  best valid loss : {self.best_valid_loss:.4f}")
        print(f"  best checkpoint : {self.out_path / 'best.pt'}")
        print(f"  final checkpoint: {final_path}")
        print(f"  wall-clock      : {total_min:.1f} min")
        print("=" * 64)
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
    parser.add_argument("--log-every", type=int, default=20,
                        help="print a loss/lr line every N optimizer steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="load data + print config/example, then exit (no model load, no training)")

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
        log_every=args.log_every,
        dry_run=args.dry_run,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
