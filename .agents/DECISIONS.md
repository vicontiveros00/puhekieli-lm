# DECISIONS

> Append-only log of non-obvious choices and *why*. Newest at top.
> Older entries compacted 2026-07-23 — kept the *why*, dropped the play-by-play.

## 2026-07-23 — First fine-tune on Qwen3-0.6B; chat.py rewritten for LoRA
- **Base = Qwen/Qwen3-0.6B** (not Qwen3.5-0.8B): the 0.8B is a *multimodal*
  `Qwen3_5ForConditionalGeneration`, not a plain causal LM — less portable and won't
  load cleanly via `AutoModelForCausalLM`. Picked the smallest solid text-only base
  for a side project. First run: 2 epochs, batch 2, max-len 512, ~197 min on M4 Pro;
  valid loss 2.2721 → **2.2056** (still dropping → more epochs may help). Adapter in
  `checkpoints/qwen3-0.6b-lora-2e-2ep/`. Output is real puhekieli (“mä”/“oon”/“sä”)
  but loose at this size — signal, not a finished translator.
- **`chat.py` rewritten:** old version loaded the LoRA checkpoint as a *full* model
  (it's only an adapter), skipped the training chat template, and used fragile
  fp16 + `device_map="auto"`. Now loads base + applies adapter via PEFT, uses the
  chat template, decodes only new tokens, strips Qwen3's empty `<think>` block, adds
  `--repetition-penalty`, and supports a no-adapter **baseline** mode for comparison.

## 2026-07-22 — finetune.py: observability pass + scheduler fix
Kept the user in the loop and fixed latent training bugs.
- **Observability (no behaviour change):** config banner; data-composition summary;
  one decoded example; per-step `tqdm` with live loss/lr; periodic log lines
  (`--log-every`); honest checkpoint feedback; final recap; `--dry-run` (load data +
  print config, then exit).
- **Scheduler fix (changes training dynamics):** old `CosineAnnealingLR(T_max=epochs)`
  stepped per *epoch* and `warmup_steps` was never applied → now
  `get_cosine_schedule_with_warmup` stepping per *optimizer step*, warmup =
  `min(warmup_steps, total_steps//10)`. Also added the missing `optimizer.zero_grad()`
  (grads had been accumulating across steps).
- **Honest LoRA failure:** narrowed swallow-all `except` → `ImportError` with a loud
  OOM warning (was silently claiming "training fully fine-tuned" on any error).

## 2026-07-07 — Project goal: EN → puhekieli (spoken), not kirjakieli
Deliberately target everyday **spoken** Finnish, not formal written Finnish — it's
the harder, more useful register. Consequences: training data must be spoken/
colloquial; eval must reward spoken features (mä/sä/me-passive/ne/contractions), not
just grammaticality (a kirjakieli BLEU reference would penalize success); no single
canonical puhekieli, so we pick a general everyday style and accept variation.

## 2026-07-07 — Data sources: subtitles + OPUS + rap + synthetic back-translation
Four complementary sources: **`opensubtitles_enfi`** (HF-streamed) = base EN→FI signal
(dialogue register); **`opus_100`** (HF-streamed) = broad mixed-domain pairs;
**`genius_rap`** (Gettomasa/JVG/Ibe/Etta/Costi) = the target Helsinki puhekieli
register but FI-only; **`rap_synthetic`** = the bridge — back-translate each real
lyric FI→EN locally, then train on (synthetic EN → real FI). FI target is always
authentic (input noise ≪ output noise), so the model learns to *produce* genuine
puhekieli. Standard low-resource MT trick; this is what actually teaches rap register.

## 2026-07-07 — Genius scraping blocked → curated seed + HF-streamed pairs
Genius gates lyric HTML behind Cloudflare Private Access Tokens (Apple hardware
attestation) — no automation beats it (lyricsgenius/curl_cffi/Playwright all 403 or
loop). So **`genius_rap` is a curated seed**: the JSON API lists song URLs only; you
paste favourite lines into `data/raw/genius_rap/<artist>.txt` (fewer but authentic,
zero arms race). Parallel corpora stream from HF (subtitles + OPUS-100) instead of
big moses zips — lighter on disk, no custom loaders.

## 2026-07-07 — Back-translation model: gpt-oss-20b default (qwen3-14b fallback)
Local LM Studio does the FI→EN back-translation. **`gpt-oss-20b`** runs best here
(MXFP4 → memory-light) and is the default. Both candidates are reasoning models that
return empty `content` unless allowed to finish thinking, so `back_translate_line`
uses a big token budget (1024), `reasoning_effort=low`, `/no_think` for qwen-style,
and strips leaked think-blocks/quotes. Swappable via `LMSTUDIO_MODEL`. Quality on
hard slang ~tied; gsoss ~11s/line vs qwen ~6s/line.

## 2026-07-07 — puhekieli eval is a heuristic, on purpose
`eval.puhekieli_score` counts spoken markers (mä/sä/ne/me-passive/stadin slangi) vs
formal ones and returns a [-1,1] lean. A cheap scoreboard, not linguistics — used
*alongside* BLEU because BLEU vs a kirjakieli reference would penalize the exact
spoken forms we want.

## 2026-07-07 — Sources are user-chosen; registry pluggable, nothing active by default
`config.py::SOURCES` registry carries a Finnish `register` + `status`
(planned/active/excluded) per source. Nothing is `active` until the user clears it —
keeps the pipeline honest about license/register and avoids scraping anything the
owner hasn't chosen.

## 2026-07-07 — Reused the solita-llm scaffold; inherited stack
Kept the template's shape (uv + Python 3.11, MPS device auto-select MPS→CUDA→CPU,
`src/<pkg>/config.py` as single source of truth, `.agents/` handoff docs, two-act
narrative) because it works and the owner knows it. 3.11 is the ML-lib sweet spot;
model sizes fit 24GB unified memory (Act 1: ~10–50M params). Only the domain changed:
company-data LM → EN→puhekieli translator; schema shifts to parallel `{en, fi, register}`.

## 2026-07-07 — Phase 1 executed: cleaned rap lyrics + synthetic back-translation
Built `genius_rap.jsonl` (6,016 puhekieli lines) and back-translated via `gpt-oss-20b`
into `rap_synthetic.jsonl`. Sample (90 pairs): FI authentic puhekieli, EN natural;
≈59% spoken-leaning (acceptable mix). Then scaled to full corpus.
