"""
Chat with a fine-tuned puhekieli model (base model + LoRA adapter).

The fine-tune saves a LoRA *adapter* (adapter_model.safetensors) under
checkpoints/<name>/best.pt/, not a full model. So we load the original base
model and apply the adapter on top. Prompts are formatted with the same chat
template used during training.

Usage:
    uv run python scripts/chat.py \
        --base Qwen/Qwen3-0.6B \
        --adapter checkpoints/qwen3-0.6b-lora-2e-2ep/best.pt \
        "Translate to Finnish: I'm going to the club tonight, are you with me?"

    # interactive (omit the prompt and it will ask):
    uv run python scripts/chat.py --base Qwen/Qwen3-0.6B --adapter checkpoints/qwen3-0.6b-lora-2e-2ep/best.pt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Chat with a puhekieli LoRA model")
    parser.add_argument("--base", type=str, required=True,
                        help="Base model id/path (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="LoRA adapter checkpoint dir (e.g. checkpoints/.../best.pt). "
                             "Omit to run the raw base model for comparison.")
    parser.add_argument("prompt", type=str, nargs="?", default="",
                        help="English text to translate (a 'Translate to Finnish:' prefix is added if missing)")
    parser.add_argument("--max-len", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="penalize repeated tokens (>1 reduces loops)")
    args = parser.parse_args()

    # Tokenizer: prefer the one saved with the adapter, else the base.
    tok_src = str(args.adapter / "tokenizer") if args.adapter else args.base
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base} (device: {device}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)

    if args.adapter:
        from peft import PeftModel
        print(f"Applying LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    else:
        print("No adapter given -> running raw base model (baseline comparison).")

    model.to(device)
    model.eval()

    user_text = args.prompt or input("Enter English text: ")
    # Match the training prompt format.
    if not user_text.lower().startswith("translate to finnish"):
        user_text = f"Translate to Finnish: {user_text}"

    messages = [{"role": "user", "content": user_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    # Decode only the newly generated tokens (skip the prompt).
    gen = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    # Qwen3 is a reasoning model; drop any empty/leftover <think> block.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    print("\nPROMPT:")
    print(f"  {user_text}")
    print("\nPUHEKIELI:")
    print(f"  {text.strip()}")


if __name__ == "__main__":
    main()
