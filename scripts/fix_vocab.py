"""
Regenerate vocabulary using HuggingFace tokenizers with proper BPE
to capture puhekieli markers (*städi, vittu, jäbä, luuliks, op*).
"""

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

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

def build_vocab(texts: list[str], vocab_size: int = 20000) -> str:
    """Build HF BPE tokenizer and return vocabulary as space-separated tokens."""
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<BOS>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer.encode_batch(texts)[0].tokens

if __name__ == "__main__":
    print("Loading FI texts...")
    from puhekieli_llm.config import CLEAN, TOKENIZED
    fi_texts = []

    for path in [CLEAN / "genius_rap.jsonl", CLEAN / "rap_synthetic.jsonl"]:
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    fi_texts.append(obj["fi"])
    print(f"Loaded {len(fi_texts)} text lines")

    puhekieli_keywords = ["städi", "vittu", "jäbä", "luuliks", "op"]
    print(f"Checking samples for puhekieli keywords: {puhekieli_keywords}")

    # Build vocab
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=20000,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<BOS>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(fi_texts, trainer=trainer)

    token_count_before = sum(len(t.split()) for t in fi_texts)
    print(f"Starting with {token_count_before} raw tokens")

    vocab_tokens = tokenizer.get_vocab()
    vocab_tokens = sorted(vocab_tokens.keys())
    vocab_tokens = [t for t in vocab_tokens if len(t) > 1]  # filter single chars

    # Save vocab
    vocab_path = TOKENIZED / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens))

    print(f"Vocabulary saved to {vocab_path}")
    print(f"Vocab size: {len(vocab_tokens)} tokens")

    # Check puhekieli coverage
    found = [kw for kw in puhekieli_keywords if kw in vocab_tokens]
    print(f"Puhekieli words in vocab: {found}")

    # Sample encodings
    print("\nSample encodings of puhekieli lines:")
    for text in fi_texts[:2]:
        enc = tokenizer.encode(text)
        print(f"  {text[:50]}")
        print(f"  → {enc.tokens[:15]}")
        break
