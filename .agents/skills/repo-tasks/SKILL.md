---
name: repo-tasks
description: Step-by-step playbooks for common tasks in the puhekieli-llm repo (running notebooks, adding deps, collecting a data source, training, committing safely). Use whenever working in this repo and you need the exact, safe procedure — especially helpful for smaller local models that benefit from explicit steps.
---

# puhekieli-llm repo tasks

Explicit, safe procedures for this repo. Always start from the repo root:
`cd /Users/victormanuel.ontiveros/repos/puhekieli-llm`

## GROUND RULES (apply to every task)
- This folder is its own git repo. NEVER `git add -A` from a parent dir.
- NO `data/` or `models/` content in git (git-ignored; double-check before commit).
- Only use data sources marked `active` in `config.py::SOURCES`. Respect licenses.
- The goal is **puhekieli** (spoken Finnish), not kirjakieli — keep data/eval aligned.
- Use `uv run ...` — never call python/pip directly.
- Keep changes minimal. Verify before committing. Update `.agents/STATUS.md`.

---

## Task: run / verify a notebook
```bash
cd /Users/victormanuel.ontiveros/repos/puhekieli-llm
uv run jupyter nbconvert --to notebook --execute --stdout notebooks/<name>.ipynb
```
If it runs clean, strip outputs before any commit:
```bash
uv run jupyter nbconvert --clear-output --inplace notebooks/<name>.ipynb
```

## Task: add a Python dependency
- Core dep: add to `[project].dependencies` in `pyproject.toml`, then `uv sync`.
- Phase-specific: add to the matching extra (`scrape` or `finetune`), then
  `uv sync --extra <name>`.
- Do NOT `pip install` into the venv directly — it won't be reproducible.

## Task: add a new data source
Read `.agents/DATA_POLICY.md` FIRST. Then:
1. Confirm the user has cleared the source. Add it to `config.py::SOURCES` with its
   `register` (want `puhekieli`) and `status: "active"`. If unsure, STOP and ask.
2. Fetch raw into `data/raw/<source>/` (untouched, so we can re-clean later).
3. Clean → one jsonl record per pair into `data/clean/<source>.jsonl`:
   `{"source": "...", "id": "...", "en": "...", "fi": "...", "register": "puhekieli"}`
4. Normalize whitespace, strip subtitle timestamps/boilerplate, dedup, scrub PII.
5. Disk is tight (~50GB). Process incrementally; don't hold the whole corpus at once.

## Task: train the tiny translator (Act 1)
- Lives in `notebooks/03_train_gpt.ipynb` (built in Phase 3).
- Device comes from `puhekieli_llm.config.DEVICE` (MPS). Don't hardcode 'cuda'/'cpu'.
- Keep params ~10–50M to fit 24GB unified memory. Save checkpoints to `models/`.

## Task: evaluate (does it produce puhekieli?)
- Don't rely on BLEU vs a kirjakieli reference alone — it penalizes spoken forms.
- Add a puhekieli-feature check: look for spoken markers (mä/sä/mä oon/me-passive
  "mennään"/ne instead of he/contractions) in outputs. Log both signals.

## Task: commit safely
```bash
cd /Users/victormanuel.ontiveros/repos/puhekieli-llm
uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb   # if nb changed
git add -A
git status --short      # CHECK: no data/ or models/ paths. If there are, STOP.
git commit -m "<clear message>"
git log --oneline -3
```

## Task: figure out what to do next
1. Read `.agents/STATUS.md` (the "Next up" section).
2. Read `.agents/PROJECT.md` for the phase table if you need the big picture.

## When something is ambiguous
Ask a short, specific question. Do not invent file paths — `ls`/read first.
See `references/MODEL_NOTES.md` for tips when running this with a small local model.
