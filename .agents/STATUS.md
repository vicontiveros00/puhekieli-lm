# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-07
**Current phase:** Phase 1 (data collection) **built** → run it, then Phase 2 (tokenizer).

## Sources chosen (2026-07-07)
Personal project, public-web fair game. Three sources, all `active` in
`config.py::SOURCES`:
- **`opensubtitles_enfi`** (`pairs`) — OPUS OpenSubtitles EN-FI, the base translation
  signal. Downloaded from OPUS moses zip (~870 MB) into `data/raw/`, stream-cleaned.
- **`genius_rap`** (`flavor`) — Finnish rap lyrics via Genius API. Artists:
  Gettomasa, JVG, Ibe, Etta, Costi. FI-only, pure Helsinki puhekieli/slang.
- **`rap_synthetic`** (`synth`) — EN→FI pairs made by back-translating rap lyrics
  with a local LLM (LM Studio). FI side is the real lyric; only EN is synthetic.

## Done
- [x] Phase 0: scaffold (retargeted solita-llm template to EN→puhekieli)
- [x] Phase 1 code + notebooks (all verified headless with graceful skips):
  - `src/puhekieli_llm/sources.py` — Genius fetch/clean + OpenSubtitles zip stream
  - `src/puhekieli_llm/synth.py` — local-LLM back-translation (resumable)
  - `src/puhekieli_llm/eval.py` — puhekieli-feature scorer (spoken vs kirjakieli)
  - `notebooks/01_collect.ipynb`, `notebooks/01b_synthesize.ipynb`
  - deps: `scrape` extra + `lyricsgenius`; new `synth` extra (`openai`)

## Next up
**Actually run Phase 1** (needs credentials/services the agent can't provide):
1. `export GENIUS_ACCESS_TOKEN=...` (https://genius.com/api-clients), launch Jupyter,
   run `01_collect.ipynb` section A → builds `data/clean/genius_rap.jsonl`.
2. Run section B with `FETCH_OPENSUBS=1` → downloads + builds
   `data/clean/opensubtitles_enfi.jsonl` (~870 MB one-time).
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
- Disk tight (~50GB): the OpenSubtitles zip is ~870 MB; `data/` is gitignored.
- Parent `~/repos` is a separate git repo — never `git add` from there.
