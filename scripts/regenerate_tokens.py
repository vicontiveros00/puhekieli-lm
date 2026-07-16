"""
Regenerate tokenization files using HF tokenizers + new vocab.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from puhekieli_llm.config import CLEAN, TOKENIZED


def load_fi_data() -> list[str]:
    """Load all cleaned puhekieli text."""
    fi_texts = []
    for path in [CLEAN / "genius_rap.jsonl", CLEAN / "rap_synthetic.jsonl"]:
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    fi_texts.append(obj["fi"])
    return fi_texts


def load_vocab() -> list[str]:
    """Load HF vocab."""
    vocab_path = TOKENIZED / "vocab.txt"
    return [t.strip() for t in vocab_path.read_text().split("\n") if t.strip()][:20000]


def tokenize_text(text: str, tokenizer: Tokenizer) -> list[str]:
    """Tokenize a text line."""
    encoded = tokenizer.encode(text)
    return encoded.tokens


if __name__ == "__main__":
    print("Loading FI texts...")
    fi_texts = load_fi_data()
    print(f"Loaded {len(fi_texts)} text lines")

    print("Loading vocabulary...")
    vocab = load_vocab()
    print(f"Loaded {len(vocab)} tokens from vocab.txt")

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokens = vocab + ["<UNK>", "<PAD>", "<SOS>", "<EOS>", "<BOS>"]
    tokenizer.add_tokens(tokens)

    # Tokenize all text
    print("\nTokenizing FI texts...")
    tokenized_fi = [tokenize_text(t, tokenizer) for t in fi_texts]

    print(f"Tokenized {len(tokenized_fi)} lines")

    # Save as multi-space token strings
    tokenizer_path = TOKENIZED / "fi_tokenized.txt"
    tokenizer_path.write_text("\n".join([" ".join(t) for t in tokenized_fi]))
    print(f"Saved tokenized FI to {tokenizer_path}")

    # Split train/valid (90/10)
    import random
    random.seed(42)
    split_idx = int(0.90 * len(fi_texts))
    random.shuffle(tokenized_fi)

    train_tokens = tokenized_fi[:split_idx]
    valid_tokens = tokenized_fi[split_idx:]

    (TOKENIZED / "rap_synthetic_train_tokens.txt").write_text("\n".join([" ".join(t) for t in train_tokens]))
    (TOKENIZED / "rap_synthetic_valid_tokens.txt").write_text("\n".join([" ".join(t) for t in valid_tokens]))

    print(f"Split: {len(train_tokens)} train, {len(valid_tokens)} valid")

    # Re-save vocab
    (TOKENIZED / "vocab.txt").write_text("\n".join(vocab))

    # Show sample
    print("\nSample tokens (FI):")
    for i, t in enumerate(tokenized_fi[:3]):
        print(f"  {i}: {t[:15]}...")
        break
