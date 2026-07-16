"""
Chat with fine-tuned puhekieli model.

Usage:
    uv run python scripts/chat.py --model checkpoints/llama3.2-3b-lora-2e4-2ep/best.pt --tokenizer checkpoints/llama3.2-3b-lora-2e4-2ep/best.pt/tokenizer "Translate to Finnish: I'm going to the club tonight, are you with me?"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Chat with puhekieli model")
    parser.add_argument("--model", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Tokenizer path (defaults to model)")
    parser.add_argument("prompt", type=str, nargs="?", default="", 
                        help="Input prompt to translate/generate")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    prompt = args.prompt or input("Enter prompt: ")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"PUHEKIELI:")
    print(text)


if __name__ == "__main__":
    main()
