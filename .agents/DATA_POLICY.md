# DATA_POLICY

> Read this before touching any data. Source registry lives in
> `src/puhekieli_llm/config.py` (`SOURCES`).

## This is a personal project — but still be careful
- **The user chooses the sources.** Don't scrape or download anything that isn't
  registered as `status: "active"` in `config.py::SOURCES`.
- **Respect licenses & terms.** Subtitle sites, forums, and corpora each have their
  own rules. Note the license per source in its registry entry / DECISIONS.md.
- **No data or model weights in git.** `data/` and `models/` are git-ignored.
- Notebook outputs are stripped before commit (keeps the repo small and clean).

## What we actually want: spoken Finnish (puhekieli)
The signal that matters is *colloquial/spoken* Finnish. Track each source's
register in the registry:
| register     | meaning                              | want it? |
|--------------|--------------------------------------|----------|
| `puhekieli`  | everyday spoken/colloquial Finnish   | ✅ yes (core) |
| `mixed`      | some spoken, some written            | ⚠️ ok, filter |
| `kirjakieli` | formal written standard              | ➖ mostly as EN-side pairs or contrast |

## Candidate source types (user decides)
- **Subtitles** (OpenSubtitles / OPUS OpenSubtitles) — naturally spoken.
- **Forums / chat** (e.g. Suomi24 corpus) — very colloquial.
- **Parallel corpora** (Tatoeba, OPUS) — lots of EN-FI pairs, mostly kirjakieli.
- **Transcripts** — podcasts, interviews, spoken-language corpora.

## Data flow (planned)
```
fetch  ->  data/raw/<source>/...             (untouched; lets us re-clean w/o re-fetch)
clean  ->  data/clean/<source>.jsonl         (parallel records, see schema below)
tokenize -> data/tokenized/*.bin             (token-id arrays for training)
```

## Record schema (parallel translation data)
```json
{"source": "...", "id": "...", "en": "...", "fi": "...", "register": "puhekieli"}
```
For FI-only spoken text (no English side yet), keep `en` empty and note we'll need
to pair/synthesize it. Always keep a `source` and `register` field so we can filter.

## Cleaning expectations
- Normalize whitespace, strip boilerplate/nav/subtitle timestamps & formatting.
- Dedup near-identical lines (subtitles repeat a lot).
- Scrub obvious PII from forum/transcript text (names, emails, phones) — raw stays
  only in `data/raw/` (git-ignored), never committed. When in doubt, exclude.
