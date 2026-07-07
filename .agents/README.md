# .agents — Agent handoff context

This directory exists so any AI agent (or future me) picking up this project can get
oriented **fast** without re-deriving decisions. Read these in order:

1. **`README.md`** (this file) — what each doc is for, golden rules
2. **`PROJECT.md`** — the goal (EN→puhekieli), the 2-act narrative
3. **`STATUS.md`** — where we are right now, what's next (UPDATE THIS as you work)
4. **`DECISIONS.md`** — why things are the way they are (append-only log)
5. **`ENVIRONMENT.md`** — machine, hardware constraints, how to run things
6. **`DATA_POLICY.md`** — data sources, registers, licensing (read before touching data)

Planning sessions live in **`plans/`** (`YYYY-MM-DD-<topic>.md`) — the "thinking"
behind decisions. `DECISIONS.md` keeps the short record of *what* was decided.

## Golden rules for agents working here

1. **No data or model weights in git.** `data/` and `models/` are git-ignored. Strip
   notebook outputs before committing (`jupyter nbconvert --clear-output --inplace`).
2. **This folder has its OWN git repo.** The parent `~/repos` is a *different* repo.
   Never run `git add -A` from a parent dir. Always `cd` here first.
3. **The user chooses data sources.** Only use sources marked `active` in
   `config.py::SOURCES`. Respect licenses (see `DATA_POLICY.md`).
4. **Use `uv`** for everything: `uv run ...`, `uv sync`. Python is pinned to 3.11.
5. **Remember the goal is puhekieli**, not kirjakieli — data and eval must reflect that.
6. **Update `STATUS.md`** when you finish a unit of work, and append to
   `DECISIONS.md` when you make a non-obvious choice.
7. **Verify notebooks headless** before declaring them done:
   `uv run jupyter nbconvert --to notebook --execute --stdout <nb>`.
