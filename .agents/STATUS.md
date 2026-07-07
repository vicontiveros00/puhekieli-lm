# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-07
**Current phase:** Phase 1 (data collection) **built** → run it, then Phase 2 (tokenizer).

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

## Next up
**Actually run Phase 1** (needs credentials/services the agent can't provide):
0. `cp .env.example .env` and fill it in (Genius token, LM Studio URL/model). It's
   gitignored and auto-loaded by `config.py` — no shell exports needed.
1. Get a Genius token (https://genius.com/api-clients) → put in `.env`, launch
   Jupyter, run `01_collect.ipynb` section A: metadata cell lists song URLs; open a
   few in your browser and paste lines into `data/raw/genius_rap/<artist>.txt` (see
   `examples/genius_rap_seed.example.txt`), then run the clean cell.
2. Run section B → streams OpenSubtitles + OPUS-100 from HF into
   `data/clean/opensubtitles_enfi.jsonl` and `data/clean/opus_100.jsonl` (cap via
   `MAX_PAIRS`; no big download).
3. Load a model in LM Studio (server on :1234), run `01b_synthesize.ipynb` — start
   with `limit=100`, eyeball, then raise. Builds `data/clean/rap_synthetic.jsonl`.
Then build **Phase 2 tokenizer** (`02_tokenizer.ipynb`) over subtitles+rap+synth.

## Watch-outs / open threads
- Back-translation model chosen: **`gpt-oss-20b`** (runs best on this box after
  relaxing LM Studio's memory guardrails; `qwen3-14b-128k` also works). Both are
  reasoning models — `synth.py` handles them (big token budget + reasoning_effort=low,
  /no_think for qwen). Default in `config.py`; LM Studio server was at
  `http://172.20.10.7:1234/v1` during testing (set `LMSTUDIO_BASE_URL` to match yours).
- No single "correct" puhekieli — target style is young Helsinki rap register.
- Eval: `eval.puhekieli_score` is a heuristic (marker counting), not linguistics.
  Use it alongside BLEU, not instead of it.
- Disk tight (~50GB): parallel corpora are streamed from HF and capped, so no giant
  local downloads; `data/` is gitignored.
- Parent `~/repos` is a separate git repo — never `git add` from there.
