#!/bin/bash
# Wrapper script to fine-tune Qwen/Qwen3-4B on puhekieli

cd /Users/victormanuel.ontiveros/repos/puhekieli-llm

uv run python scripts/finetune.py \
  --model Qwen/Qwen3-4B \
  --fi-data data/clean/genius_rap.jsonl \
  --fi-en data/clean/rap_synthetic.jsonl \
  --out checkpoints/qwen3-4b-lora-2e-2ep \
  --epochs 2 \
  --batch 2 \
  --max-len 512
