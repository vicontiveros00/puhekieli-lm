# ENVIRONMENT

## Machine
- **Apple M4 Pro**, **24 GB** unified memory, 12 CPU cores
- macOS, Apple Silicon (arm64)
- Disk: ~50 GB free — **tight**. Process data incrementally; don't hoard the full
  corpus on disk.

## Acceleration
- PyTorch uses **MPS** (Apple Metal). `torch.backends.mps.is_available() == True`.
- `puhekieli_llm.config.DEVICE` auto-selects MPS → CUDA → CPU.
- Keep model + batch within unified memory. Act 1 target: ~10–50M params.

## Toolchain
- **uv** — env + deps. Python **3.11** (pinned via `.python-version`).
- torch (MPS), tokenizers, datasets, jupyterlab, matplotlib, pandas, pydantic.
- `brew` available. `git-lfs` not needed yet.
- Local inference: **LM Studio** (OpenAI-compatible server on `:1234`), if used.

## How to run things
```bash
cd ~/repos/puhekieli-llm         # ALWAYS cd here first (git + paths)
uv sync                          # base ML stack
uv sync --extra scrape           # + data-collection deps (Phase 1)
uv sync --extra finetune         # + transformers/peft/sacrebleu (Phase 5)
uv run jupyter lab               # open notebooks/
uv run python -c "from puhekieli_llm.config import summary; print(summary())"

# verify a notebook headless before declaring it done:
uv run jupyter nbconvert --to notebook --execute --stdout notebooks/<nb>.ipynb

# strip outputs before committing:
uv run jupyter nbconvert --clear-output --inplace notebooks/<nb>.ipynb
```

## Git
- This folder is its OWN repo. Parent `~/repos` is a *different* repo — don't touch it.
- No remote configured yet. Push from THIS dir only once one is added.
