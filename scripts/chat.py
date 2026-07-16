"""
Chat interface for the fine-tuned puhekieli model.

Usage:
    uv run python scripts/chat.py --model checkpoints/llama3.2-3b-lora-2e4-2ep/best.pt "Translate to Finnish: I'm going to the club tonight, are you with me?"
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with puhekieli model")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to tokenizer (defaults to model/checkpoint)")
    parser.add_argument("prompt", type=str, nargs="?", default="",
                        help="Input prompt to translate/generate")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="K/warmup")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="K/s bootstrap")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    prompt = args.prompt or input("Enter prompt: ")

    # Tokenize and generate
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"PUHEKIELI OUTPUT:")
    print(output_text)


if __name__ == "__main__":
    main()
