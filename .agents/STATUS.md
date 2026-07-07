# STATUS

> Keep this current. It's the first thing the next agent reads to know where to start.

**Last updated:** 2026-07-07
**Current phase:** Phase 0 complete → Phase 1 (data collection) — **user picks sources next.**

## Done
- [x] Phase 0: project scaffold (mirrors the solita-llm template, retargeted to
      EN→puhekieli Finnish translation)
  - uv project, Python 3.11, torch on MPS
  - `src/puhekieli_llm/config.py`: paths, device, task langs, `SOURCES` registry
  - `notebooks/00_setup.ipynb` (env check + puhekieli vs kirjakieli intro)
  - `.gitignore` excludes all data/ and models/
  - dedicated git repo for this folder
- [x] `.agents/` handoff docs + weak-local-model support (`AGENTS.md`,
      `.pi/prompts/`, `/skill:repo-tasks`, `pi.dev/README.md`)

## Next up — Phase 1 (data collection)
**Blocked on the user choosing sources.** We need EN→puhekieli signal. Candidate
source types (user decides, then register in `config.py::SOURCES` as `active`):
- **Subtitles** (movies/TV) — naturally spoken register. Watch licensing.
- **Forums / chat** (e.g. Suomi24) — very colloquial. Watch scraping terms + PII.
- **Parallel corpora** (Tatoeba, OPUS) — lots of EN-FI pairs but mostly kirjakieli;
  useful as a base to then bias toward puhekieli.
- **Transcripts** of spoken Finnish (podcasts, interviews).

Open questions for the user:
1. Which sources are you comfortable using (license/ethics)?
2. Do you have EN↔FI *parallel* data, or FI-only spoken text (then we synthesize EN)?
3. Target a specific dialect/style, or general everyday spoken Finnish?

Once sources are chosen: build a **streaming/incremental** collector that writes
normalized records to `data/clean/*.jsonl`. Suggested schema for parallel data:
`{"source": "...", "id": "...", "en": "...", "fi": "...", "register": "puhekieli"}`
Keep raw fetches in `data/raw/` so we can re-clean without re-fetching.

## Watch-outs / open threads
- No single "correct" puhekieli — decide the target style and note it in DECISIONS.
- Eval can't just be BLEU vs a kirjakieli reference; add a puhekieli-feature check.
- Disk is tight (~50GB) — process incrementally; don't hoard the full corpus.
- Parent `~/repos` is a separate git repo — never `git add` from there.
