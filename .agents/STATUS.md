# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-22
**Current phase:** Phase 5 (LoRA fine-tune) in progress — pipeline built, no completed run yet.

## TL;DR for the next agent
Data is collected & scaled. The custom-tokenizer path (Phase 2) was tried then
dropped in favour of the base model's HF `AutoTokenizer`. The LoRA fine-tune
pipeline (`scripts/finetune.py`) is built and pointed at **Qwen/Qwen3-4B** (see
the fine-tune command in `README.md`), but **no fine-tune has completed** — all
three `checkpoints/*` dirs are empty. `scripts/finetune.py` has substantial
uncommitted changes (causal-LM rewrite).
**Next concrete action:** run the Phase 5 fine-tune command (README), confirm it
produces weights, then commit.

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

## Next up
**Phase 5 fine-tune has NOT completed yet.** Concrete steps:

1. Commit the pending work first (review the diff): `scripts/finetune.py` was
   reworked from seq2seq → causal-LM (chat-template formatting, `Translate to
   Finnish:` prompts, `DataCollatorForLanguageModeling`).
2. Run the Phase 5 fine-tune command (see `README.md`) and confirm it actually
   writes weights — the three `checkpoints/*` dirs (`qwen2.5-1.5b-lora-2e-2ep`,
   `llama3.2-3b-lora-2e-2ep`, `qwen3-4b-lora-2e-2ep`) are all **empty** (runs were
   set up but produced nothing).
3. Once a checkpoint exists, evaluate with `scripts/chat.py` + `eval.puhekieli_score`
   (spoken-lean) alongside BLEU. Compare base vs LoRA output register.
4. Phase 3/4 (train-from-scratch tiny GPT + generate/eval notebooks) remain TODO —
   Act 1 was effectively skipped in favour of going straight to Act 2 LoRA. Decide
   whether to backfill Act 1 or keep focus on the fine-tune.

## Uncommitted / in-flight (as of 2026-07-22)
- `checkpoints/{qwen2.5-1.5b,llama3.2-3b,qwen3-4b}-lora-2e-2ep/` — empty dirs (no weights)

## Watch-outs / open threads
- Back-translation model: **`gpt-oss-20b`** runs best on this box after relaxing LM Studio's memory guardrails (still exploring stability). Alternative `qwen3-14b-128k` also works with adjusted token budget.
- No single "correct" puhekieli — target style is young Helsinki rap register.
- Eval: `eval.puhekieli_score` is heuristic (marker counting), not linguistics.
  Use alongside BLEU/SacreBLEU, not instead of it.
- Disk tight (~50GB): parallel corpora are streamed from HF and capped; `data/` is gitignored.
- Parent `~/repos` is a separate git repo — never `git add` from there.
- Keep notebook outputs clean (`uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before commit).
