# DECISIONS

> Append-only log of non-obvious choices and *why*. Newest at top.

## 2026-07-07 — Project goal: EN → puhekieli (spoken Finnish), not kirjakieli
Personal holiday spin-off of the solita-llm template. Standard MT/LLMs translate
EN→FI into formal written Finnish (kirjakieli). This project deliberately targets
**puhekieli** (everyday spoken Finnish) because that's the harder, more interesting,
and more personally useful register. Consequences that shape everything:
- Training data must be *spoken/colloquial* Finnish (subtitles, forums, transcripts).
- Eval must reward spoken features (mä/sä/me-passive/ne/contractions), not just
  grammaticality — a kirjakieli-only BLEU reference would penalize success.
- There's no single canonical puhekieli; we pick a general everyday style and accept
  variation.

## 2026-07-07 — Sources are user-chosen; registry is pluggable, nothing active
This is a personal project, so no corporate sensitivity tiers. Instead `config.py`
has a `SOURCES` registry where each source carries a Finnish `register` and a
`status` (planned/active/excluded). Nothing is `active` until the user clears it.
Rationale: keep the data pipeline honest about license/register and avoid scraping
anything the owner hasn't chosen.

## 2026-07-07 — Reused the solita-llm scaffold wholesale
Kept the same shape (uv + Python 3.11, MPS device, `src/<pkg>/config.py` as single
source of truth, `.agents/` handoff docs, `.pi/prompts/`, `/skill:repo-tasks`,
two-act narrative) because it works and the owner already knows it. Only the domain
changed: company-data LM → EN→puhekieli translator. Data schema shifts from single
`text` docs to parallel `{en, fi, register}` records.

## 2026-07-07 — Python 3.11 + uv + MPS (inherited)
Same reasoning as the template: 3.11 is the ML-lib sweet spot; uv for reproducible
envs; torch on Apple Metal (MPS), model sizes chosen to fit 24GB unified memory
(Act 1: ~10–50M params). Device auto-selects MPS→CUDA→CPU.
