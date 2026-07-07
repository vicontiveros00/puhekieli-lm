---
description: Verify a notebook runs headless, then strip its outputs
argument-hint: "<notebook-name>"
---
Notebook: ${1:-00_setup}

Run exactly these from the repo root, in order. Stop and report if any step errors.
```bash
cd /Users/victormanuel.ontiveros/repos/puhekieli-llm
uv run jupyter nbconvert --to notebook --execute --stdout notebooks/${1:-00_setup}.ipynb
uv run jupyter nbconvert --clear-output --inplace notebooks/${1:-00_setup}.ipynb
```
Then confirm: did it execute without errors, and are outputs now stripped?
