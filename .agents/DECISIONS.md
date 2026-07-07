# DECISIONS

> Append-only log of non-obvious choices and *why*. Newest at top.

## 2026-07-07 — Back-translation model: gpt-oss-20b default (qwen3-14b fallback)
Tested the LM Studio server (172.20.10.7:1234) for the FI→EN back-translation step.
After relaxing LM Studio's memory guardrails, **`gpt-oss-20b`** loads and runs best
on this box (MXFP4 quant → memory-light despite "20b"). **Chosen as default.**
- Both candidates are reasoning models and return empty `content` unless allowed to
  finish thinking. Fixes in `back_translate_line`:
  - big token budget (`_MAX_TOKENS=1024`) so gpt-oss reaches its answer
    (was `finish_reason=length` → empty at 256).
  - `reasoning_effort=low` to keep it brief.
  - qwen-style models additionally get `/no_think` appended (skips reasoning; faster).
  - `_strip_output` removes leaked `<think>` blocks and smart/plain wrapping quotes.
- Quality (hard Helsinki slang) is ~tied: gpt-oss phrasing slightly more natural
  ("heading out for a gig", keeps "Stadi"), qwen a touch more literal. Both miss
  *friidu*=girl. gpt-oss ~11s/line (reasoning budget), qwen ~6s/line.
- Swappable via `LMSTUDIO_MODEL` env; default in `config.py`. LM Studio unloads on
  idle → first call reloads (brief lag).

## 2026-07-07 — Data sources: OpenSubtitles + Genius rap + synthetic back-translation
Personal project, public web is fair game. Three complementary sources:
- **`opensubtitles_enfi`** (OPUS OpenSubtitles EN-FI) = the base translation signal.
  Real EN→FI pairs, dialogue register (leans colloquial but translator-normalized).
- **`genius_rap`** (Gettomasa, JVG, Ibe, Etta, Costi via Genius API) = the target
  register — young Helsinki puhekieli/slang — but FI-only, so it can't train a
  translator alone.
- **`rap_synthetic`** = the bridge. We back-translate each real lyric FI→EN with a
  local LLM (LM Studio), then train on (synthetic EN → real FI). The FI target is
  always authentic, so the model learns to *produce* genuine puhekieli; only the
  English input is synthetic (input noise ≪ output noise). Standard low-resource MT
  trick, and the thing that actually teaches rap-register output (vs. tokenizer/eval
  which only make slang representable/measurable).
How they combine: subtitles give broad ability; synthetic rap pairs push output
toward hard Helsinki register; rap lyrics also seed the tokenizer + puhekieli eval.

## 2026-07-07 — Fetch OpenSubtitles straight from OPUS, not HF datasets
The HuggingFace `open_subtitles` loader is script-based and modern `datasets`
refuses it ("Dataset scripts are no longer supported"). So we download the OPUS
moses zip directly (`OPUS-OpenSubtitles/v2018/moses/en-fi.txt.zip`, ~870 MB) into
`data/raw/` and stream the two aligned members out of the zip without extracting.
Capped via `max_pairs` for a laptop-friendly demo; raw zip cached for re-cleaning.

## 2026-07-07 — puhekieli eval is a heuristic, on purpose
`eval.puhekieli_score` counts spoken markers (mä/sä/ne/me-passive/stadin slangi) vs
formal ones (minä/sä/hän/olen/menemme) and returns a [-1,1] lean. It's a cheap
scoreboard, not linguistics — used *alongside* BLEU because BLEU vs a kirjakieli
reference would penalize the exact spoken forms we want.

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
