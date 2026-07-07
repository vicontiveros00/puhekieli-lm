# puhekieli-llm

**Teaching a small LLM to translate English → *spoken* Finnish (puhekieli)** — a
personal holiday experiment. Off-the-shelf models translate EN→FI into the formal
written standard (**kirjakieli**). The goal here is the register people actually
*speak*: **puhekieli**. Let's see how far a from-scratch + LoRA model on an M4 Pro
can get.

Everything runs **locally** (Apple Silicon / MPS).

---

## The epic (short version)

The whole story in punchy points — the *why* behind the choices, no fluff.

- **Goal:** translate English into how Finns actually *talk*, not textbook Finnish.
- **Why it's hard:** models train on written text, so EN→FI comes out formal.
- **From scratch first:** build a tiny GPT end-to-end for the bragging rights.
- **Then LoRA:** fine-tune a real small model — the realistic path to usable output.
- **Sources are the trick:** rap lyrics are perfect slang but wrong shape (no English).
- **OpenSubtitles + OPUS-100 = the base:** real EN→FI pairs, dialogue + broad domains.
- **Rap lyrics = the flavor:** Gettomasa/JVG/Ibe/Etta/Costi — pure Helsinki slang.
  Genius blocks scraping (Cloudflare), so these are a small *hand-curated* seed.
- **Back-translation = the bridge:** a local LLM turns rap lines into English, so we
  get EN→FI pairs where the Finnish side is a *real* lyric.
- **Key insight:** only the English input is synthetic; the puhekieli target is genuine.
- **Eval is custom:** BLEU would punish slang, so we also score "how spoken is it?".
- **All local:** no data leaves the laptop; LM Studio does the back-translation.
- **It's a toy:** personal holiday project — for learning and bragging, not production.

---

## Kirjakieli vs. puhekieli (the whole point)

| English | kirjakieli (what models give you) | puhekieli (what we want) |
|---------|-----------------------------------|--------------------------|
| I am    | minä olen                         | mä oon                   |
| we go   | me menemme                        | me mennään               |
| it is   | se on                             | se on / se o             |
| they    | he                                | ne                       |
| don't know | en tiedä                       | emmä tiiä / mä en tiiä   |

Standard MT is trained on written text, so it defaults to kirjakieli. To hit
puhekieli we need **spoken/colloquial training data** and **evaluation that checks
for spoken features**, not just "grammatically correct Finnish".

---

## The two acts

### Act 1 — From scratch
Train a small GPT (~10–50M params) end-to-end: collect EN→puhekieli data → clean →
train a custom BPE tokenizer → implement & train a transformer → translate.
Small/cute but demonstrably *ours*, and runs on an M4 Pro.

### Act 2 — Make it actually work
LoRA fine-tune a real pretrained small model (Llama 3.2 / Qwen 2.5, or a
Finnish-capable base like Poro) on EN→puhekieli pairs. This is the realistic path
to something usable.

---

## Phases

| Phase | Notebook | What |
|-------|----------|------|
| 0 ✅ | `00_setup.ipynb` | env + MPS check, project layout, the puhekieli goal |
| 1 | `01_collect.ipynb` | gather spoken-Finnish + EN↔FI parallel data → `data/raw` |
| 2 | `02_tokenizer.ipynb` | train & explore a custom BPE tokenizer |
| 3 | `03_train_gpt.ipynb` | build + train tiny seq2seq/GPT translator |
| 4 | `04_generate_eval.ipynb` | translate, loss curves, puhekieli eval |
| 5 | `05_finetune_lora.ipynb` | LoRA fine-tune a real model on EN→puhekieli |
| 6 | demo writeup | write up how far it got |

---

## Data sources

**You choose the sources.** Register each in `src/puhekieli_llm/config.py`
(`SOURCES`) with its Finnish register and a status. We want *spoken/colloquial*
Finnish: subtitles, forum posts, transcripts, chat-style text. Respect each
source's license. See `.agents/DATA_POLICY.md`.

---

## Setup

```bash
uv sync                      # core ML stack (torch, tokenizers, datasets, jupyter)
uv sync --extra scrape       # + data-collection tools (Phase 1)
uv sync --extra finetune     # + transformers/peft + sacrebleu (Phase 5)
uv run jupyter lab           # open notebooks/
```

Python pinned to **3.11**. Device auto-selects MPS → CUDA → CPU.

## Project layout

```
data/raw/        untouched collected sources             (git-ignored)
data/clean/      normalized, deduped parallel text (jsonl)(git-ignored)
data/tokenized/  token-id arrays (.bin) for training     (git-ignored)
models/          tokenizer + checkpoints                 (git-ignored)
src/puhekieli_llm/  shared code (config, paths, device, sources)
notebooks/       the narrative, phase by phase
```

> **No data or model weights are committed.** See `.gitignore`.

---

## For agents / future me

See **`.agents/`**: project goal (`PROJECT.md`), current state (`STATUS.md`),
decision log (`DECISIONS.md`), environment (`ENVIRONMENT.md`), data notes
(`DATA_POLICY.md`). Start with `.agents/README.md`.

## Working with local models (token-saving)

Tuned for weaker local models. See **`pi.dev/README.md`**.
- `AGENTS.md` — tight rules pi auto-loads
- `.pi/prompts/` — canned commands: `/status`, `/nb <name>`, `/commit <msg>`
- `/skill:repo-tasks` — step-by-step playbooks
