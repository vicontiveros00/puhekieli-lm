# PROJECT

## What we're building
A **personal holiday experiment**: train an LLM to translate **English → spoken
Finnish (puhekieli)**, narrated through Jupyter notebooks. Off-the-shelf models
default to formal written Finnish (kirjakieli); the goal is the register people
actually *speak*. The point is to learn and to see how far a local model on an
M4 Pro can get — not to ship a product.

## Who it's for
- Me (Victor), on holiday, tinkering.
- Anyone curious who reads the notebooks later.

## The puhekieli problem (the core idea)
Standard MT/LLMs are trained mostly on written text, so EN→FI comes out as
kirjakieli: "minä olen", "me menemme", "he". Real spoken Finnish contracts and
shifts: "mä oon", "me mennään", "ne". Hitting puhekieli requires:
1. **Spoken/colloquial training data** (subtitles, forums, transcripts, chat).
2. **Eval that rewards spoken features**, not just grammatical correctness.
3. Accepting there's *no single* puhekieli — dialects/registers vary. We aim for a
   plausible, natural everyday spoken style, not a canonical answer.

## The two acts (the narrative spine)

### Act 1 — From scratch
Train a small GPT / seq2seq (~10–50M params) end-to-end: collect → clean → custom
BPE tokenizer → transformer → translate. Small/dumb but demonstrably *ours*.

### Act 2 — Make it actually work
LoRA fine-tune a real pretrained small model (Llama 3.2 / Qwen 2.5, or a
Finnish-capable base) on EN→puhekieli pairs. The realistic path to usable output.

## Phase plan
| Phase | Notebook | What | State |
|-------|----------|------|-------|
| 0 | `00_setup.ipynb` | env + MPS check, layout, the puhekieli goal | ✅ done |
| 1 | `01_collect.ipynb` | gather spoken-FI + EN↔FI data → data/raw | ⏳ next |
| 2 | `02_tokenizer.ipynb` | train & explore custom BPE tokenizer | todo |
| 3 | `03_train_gpt.ipynb` | build + train tiny translator | todo |
| 4 | `04_generate_eval.ipynb` | translate, loss curves, puhekieli eval | todo |
| 5 | `05_finetune_lora.ipynb` | LoRA fine-tune a real model | todo |
| 6 | demo writeup | write up how far it got | todo |

## Owner
Victor. Personal project; sources chosen by the owner as they go.
