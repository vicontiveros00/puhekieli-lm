# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-07
**Current phase:** Phase 1 complete → Phase 2 (tokenizer) next.

## Sources chosen (2026-07-07)

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
- [x] Phase 1 executed:
  - Cleaned `data/clean/genius_rap.jsonl` (6,016 puhekieli rap lyric lines)
  - Generated `data/clean/rap_synthetic.jsonl` (90 test pairs via back-translation)
  - Translation quality verified (FI: authentic puhekieli; EN: natural English)
  - Puhekieli spoken-leaning rate: ≈59% on test sample (acceptable, indicates a mix of formal drift with authentic rap register)

## Next up
**Phase 1 is complete**. Now:

1. Run `notebooks/01b_synthesize.ipynb` with `limit` raised to target corpus size
   (e.g., ~6000 lines) to build full synthetic parallel dataset.
2. Proceed to **Phase 2: Tokenizer** (`02_tokenizer.ipynb`) over:
   - `genius_rap.jsonl` (FI-only flavor)
   - `rap_synthetic.jsonl` (parallel EN-FI)
3. Fine-tuning preparation: review `puhekieli` register definition, consider
   adding additional spoken-sources (optional).

If expanding to full Phase 1+2 for dataset variety, uncomment `opensubtitles_enfi` and `opus_100` in `config.py::SOURCES.run` and run `notebooks/01_collect.ipynb`.

## Watch-outs / open threads
- Back-translation model: **`gpt-oss-20b`** runs best on this box after relaxing LM Studio's memory guardrails (still exploring stability). Alternative `qwen3-14b-128k` also works with adjusted token budget.
- No single "correct" puhekieli — target style is young Helsinki rap register.
- Eval: `eval.puhekieli_score` is heuristic (marker counting), not linguistics.
  Use alongside BLEU/SacreBLEU, not instead of it.
- Disk tight (~50GB): parallel corpora are streamed from HF and capped; `data/` is gitignored.
- Parent `~/repos` is a separate git repo — never `git add` from there.
- Keep notebook outputs clean (`uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before commit).
