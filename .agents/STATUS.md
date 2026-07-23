# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-23
**Current phase:** Phase 5 (LoRA fine-tune) — **first fine-tune completed** on
Qwen3-0.6B; inference working. Now in the try-it/evaluate loop.

## TL;DR for the next agent
Data is collected & scaled. The custom-tokenizer path (Phase 2) was tried then
dropped in favour of the base model's HF `AutoTokenizer`. The LoRA fine-tune
pipeline (`scripts/finetune.py`) is built, observable, and **has now produced a
real checkpoint**: `checkpoints/qwen3-0.6b-lora-2e-2ep/` (base **Qwen/Qwen3-0.6B**,
2 epochs, batch 2, max-len 512, ~197 min on M4 Pro; best valid loss **2.2056**,
down from 2.2721 at epoch 1). `scripts/chat.py` was rewritten to load base + LoRA
adapter with the training chat template and now generates puhekieli
(“mä”/“oon”/“sä”), though output is loose at this tiny size.
**Next concrete action:** commit `chat.py`, then evaluate the checkpoint more
systematically (`eval.puhekieli_score` + BLEU, base-vs-LoRA), and/or try more
epochs since valid loss was still dropping.

## Sources chosen (2026-07-07, still current)

Personal project, public-web fair game. Four sources, all `active` in
`config.py::SOURCES`:
- **`opensubtitles_enfi`** (`pairs`) — OpenSubtitles EN-FI, base translation signal.
  Streamed from HF (`sentence-transformers/parallel-sentences-opensubtitles`), capped.
- **`opus_100`** (`pairs`) — OPUS-100 EN-FI, broad mixed-domain pairs. Streamed from HF
  (`Helsinki-NLP/opus-100`), capped.
- **`genius_rap`** (`flavor`) — Finnish rap lyrics, **curated seed**. Genius blocks
  scraping (Cloudflare PAT), so we use the API for song URLs only and paste lines into
  `data/raw/genius_rap/<artist>.txt`. Artists: Gettomasa, JVG, Ibe, Etta, Costi.
- **`rap_synthetic`** (`synth`) — EN→FI pairs made by back-translating rap lyrics
  with a local LLM (LM Studio). FI side is the real lyric; only EN is synthetic.

## Done
- [x] Phase 0: scaffold (retargeted solita-llm template to EN→puhekieli)
- [x] Phase 1 code + notebooks (all verified headless with graceful skips):
  - `src/puhekieli_llm/sources.py` — Genius API metadata + curated-seed cleaner;
    OpenSubtitles + OPUS-100 streamed from HF
  - `src/puhekieli_llm/synth.py` — local-LLM back-translation (resumable)
  - `src/puhekieli_llm/eval.py` — puhekieli-feature scorer (spoken vs kirjakieli)
  - `notebooks/01_collect.ipynb`, `notebooks/01b_synthesize.ipynb`
  - deps: `scrape` extra (httpx/bs4/trafilatura); `synth` extra (`openai`)
- [x] Phase 1 executed **and scaled to full corpus**. `data/clean/` now holds:
  - `genius_rap.jsonl` — 6,016 authentic puhekieli rap lines (FI-only flavor)
  - `rap_synthetic.jsonl` — 5,480 EN→FI pairs (real lyric FI + back-translated EN)
  - `opensubtitles_enfi.jsonl` — 50,000 pairs (HF-streamed base signal)
  - `opus_100.jsonl` — 50,000 pairs (HF-streamed broad mixed-domain)
  - Back-translation quality verified (FI authentic puhekieli; EN natural English);
    ≈59% spoken-leaning on sample.
- [x] Phase 2 (tokenizer): tried a custom BPE tokenizer
  (`notebooks/02_tokenizer.ipynb`, token files under `data/tokenized/`), then
  **dropped it** — `finetune.py` and `chat.py` now use the base model's HF
  `AutoTokenizer` (commit 41872bc). Custom vocab no longer on the critical path.
- [x] Phase 5 pipeline built: `scripts/finetune.py` (HF `AutoModelForCausalLM` +
  PEFT LoRA on `q_proj`/`v_proj`, bf16 on MPS), `scripts/chat.py` (inference).
  Fine-tune command documented in `README.md` (Qwen/Qwen3-4B, 2 epochs, batch 2,
  max-len 512). `finetune` extra added (`transformers`, `peft`); confirmed
  importable on M4 Pro.
- [x] `finetune.py` observability + scheduler fix (2026-07-22): config banner, data
  summary, decoded example, per-step tqdm w/ loss+lr, `--log-every`, `--dry-run`,
  checkpoint feedback, final recap; fixed warmup/scheduler no-op (now per-step cosine
  w/ warmup) + missing `zero_grad`. Smoke-tested (loss 4.78→2.58, adapter saved).
  See DECISIONS 2026-07-22.
- [x] **First real fine-tune completed (2026-07-23)** on base **Qwen/Qwen3-0.6B**
  (chosen over Qwen3.5-0.8B for portability — 0.8B is a multimodal
  `Qwen3_5ForConditionalGeneration`, not a plain causal LM). 2 epochs, batch 2,
  max-len 512, ~197 min on M4 Pro. Valid loss 2.2721 (ep1) → **2.2056** (ep2);
  adapter saved to `checkpoints/qwen3-0.6b-lora-2e-2ep/{best,final}.pt/`. Valid loss
  was still falling → more epochs may help.
- [x] **`scripts/chat.py` rewritten (2026-07-23, uncommitted)**: old version loaded
  the LoRA checkpoint as a full model (wrong — it's an adapter), skipped the chat
  template, and used fragile fp16 + `device_map="auto"`. Now loads base + applies
  the adapter via PEFT, formats prompts with the training chat template, decodes
  only new tokens, strips Qwen3's empty `<think>` block, adds `--repetition-penalty`,
  and supports a no-adapter **baseline** mode. Verified generating puhekieli
  (“Mä tuun”/“sä oot”) — loose/driftly at 0.6B, as expected.

## Next up
**A fine-tune has completed; the loop is now evaluate / iterate.** Concrete steps:

1. Commit the pending work: `scripts/chat.py` rewrite (base + LoRA adapter, chat
   template, baseline mode) and `TEST_MODEL.txt` (copy-paste test commands).
2. Evaluate `checkpoints/qwen3-0.6b-lora-2e-2ep/best.pt` more systematically:
   run `eval.puhekieli_score` (spoken-lean) alongside BLEU/SacreBLEU on a held-out
   set, and compare **base-vs-LoRA** register (`chat.py` without `--adapter` is the
   baseline). Loss alone (2.21) doesn't tell you if the register is right.
3. Consider **more epochs** (valid loss was still dropping at ep2) and/or a larger
   base (Qwen3-4B in the README command) now that the pipeline is proven end-to-end.
4. Phase 3/4 (train-from-scratch tiny GPT + generate/eval notebooks) remain TODO —
   Act 1 was effectively skipped in favour of going straight to Act 2 LoRA. Decide
   whether to backfill Act 1 or keep focus on the fine-tune.

## Uncommitted / in-flight (as of 2026-07-23)
- `scripts/chat.py` — rewritten (base + adapter, chat template, baseline mode); tested
- `TEST_MODEL.txt` (project root) — copy-paste test commands
- `checkpoints/qwen3-0.6b-lora-2e-2ep/` — **has weights** (git-ignored, not committed)
- Other `checkpoints/{qwen2.5-1.5b,llama3.2-3b,qwen3-4b}-lora-2e-2ep/` — still empty

## Watch-outs / open threads
- Back-translation model: **`gpt-oss-20b`** runs best on this box after relaxing LM Studio's memory guardrails (still exploring stability). Alternative `qwen3-14b-128k` also works with adjusted token budget.
- No single "correct" puhekieli — target style is young Helsinki rap register.
- Eval: `eval.puhekieli_score` is heuristic (marker counting), not linguistics.
  Use alongside BLEU/SacreBLEU, not instead of it.
- Disk tight (~50GB): parallel corpora are streamed from HF and capped; `data/` is gitignored.
- Parent `~/repos` is a separate git repo — never `git add` from there.
- Keep notebook outputs clean (`uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before commit).
