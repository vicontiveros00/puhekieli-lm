# AGENTS.md — puhekieli-llm

> Read this fully. Keep replies short. Do exactly what's asked, nothing more.

## What this repo is
A personal holiday project: training a small LLM (from scratch, then LoRA on a real
model) to translate **English → spoken/colloquial Finnish (puhekieli)** — NOT the
formal written standard (kirjakieli) that off-the-shelf models default to. It's an
experiment; the fun is seeing how far a local model on an M4 Pro can get. Notebooks
tell the story. Everything runs LOCALLY (MPS). See `.agents/PROJECT.md`.

## CRITICAL RULES (never break these)
1. `data/` and `models/` are git-ignored — don't add them (they're large/regenerable).
2. This folder is its OWN git repo. NEVER run `git add -A` from a parent dir.
   Always `cd /Users/victormanuel.ontiveros/repos/puhekieli-llm` first.
3. Respect data-source licenses. Only use sources the user has cleared. See
   `.agents/DATA_POLICY.md` — the user decides sources; don't scrape blindly.
4. Use `uv` for everything. Python is 3.11. Don't pip-install globally.
5. Strip notebook outputs before committing:
   `uv run jupyter nbconvert --clear-output --inplace notebooks/<nb>.ipynb`

## The puhekieli angle (why this project exists)
Standard MT and LLMs translate EN→FI into **kirjakieli** (e.g. "minä olen", "me
menemme"). The goal here is **puhekieli**: how Finns actually speak (e.g. "mä oon",
"me mennään", dropped/contracted forms, spoken pronouns, colloquial vocab). So the
training data must be *spoken/colloquial* Finnish, and eval must check for
puhekieli features, not just "correct Finnish".

## Common commands (copy these exactly)
```bash
cd /Users/victormanuel.ontiveros/repos/puhekieli-llm   # always first
uv sync                          # base deps
uv sync --extra scrape           # data-collection deps (Phase 1)
uv sync --extra finetune         # fine-tune deps (Phase 5)
uv run jupyter lab               # open notebooks
uv run python -c "from puhekieli_llm.config import summary; print(summary())"
# run a notebook headless to verify:
uv run jupyter nbconvert --to notebook --execute --stdout notebooks/<nb>.ipynb
```

## Where things are
- shared code + paths + device: `src/puhekieli_llm/config.py`
- notebooks (the narrative): `notebooks/`
- handoff context: `.agents/` (PROJECT, STATUS, DECISIONS, ENVIRONMENT, DATA_POLICY)
- weak-local-model help: `pi.dev/` and `.pi/prompts/` and `/skill:repo-tasks`

## How to work
1. Check `.agents/STATUS.md` to see what's next.
2. Make the smallest change that does the job.
3. Verify it runs (headless for notebooks).
4. Update `.agents/STATUS.md`; append to `.agents/DECISIONS.md` if you made a choice.
5. Commit from THIS dir with a clear message.

## If unsure
Ask a short question instead of guessing. Don't invent file paths — list/read first.
The user is choosing the training-data sources themselves — don't assume them.
