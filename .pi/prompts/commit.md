---
description: Safely commit work in this repo (never from parent dir)
argument-hint: "<message>"
---
Commit the current work. Follow EXACTLY:
1. `cd /Users/victormanuel.ontiveros/repos/puhekieli-llm` (MUST be this dir, not a parent)
2. Strip notebook outputs if any notebooks changed:
   `uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
3. Check nothing from `data/` or `models/` is staged:
   `git add -A && git status --short`
   If you see any `data/` or `models/` path staged, STOP and tell me — do not commit.
4. Commit: `git commit -m "${@:-update}"`
5. Show `git log --oneline -3`.

Reminder: `~/repos` is a DIFFERENT git repo. Never operate git from there.
