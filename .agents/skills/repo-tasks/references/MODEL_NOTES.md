# Notes for small local models (Qwen3 14B, gpt-oss-20b, etc.)

These models are capable but drift more than frontier models. Tips so they stay
on track in this repo:

## Behaviors to watch for
- **Path hallucination.** They invent file paths. Always `ls`/read before editing.
- **Forgetting cwd.** They forget this is its own git repo. The `commit` prompt
  template and this skill repeat the `cd` every time on purpose — keep it that way.
- **Over-editing.** They rewrite whole files. Prefer small, targeted edits.
- **Skipping verification.** They claim a notebook "works" without running it.
  Always run headless (`nbconvert --execute`) before believing it.
- **Forgetting the puhekieli goal.** They default to formal Finnish. Remind them:
  we want *spoken* Finnish, and eval must check for spoken features.
- **Long-context drift.** Keep prompts short. Use `/status`, `/nb`, `/commit`.

## How to drive them efficiently (saves tokens)
- Start a session with `/status` to load only what's needed.
- Use `/skill:repo-tasks` to force-load the playbook when they ignore the
  auto-loaded description.
- Give ONE task at a time. Don't batch multiple phases.
- Ask for a 1-line plan before they act, then let them execute.

## Local inference
Local models can run via **LM Studio** (OpenAI-compatible). A pi provider setup is
not documented here yet — add it when a task needs it. Ref: pi docs `docs/models.md`,
`docs/custom-provider.md`.

## Good first tasks for a weak model here
- Run/verify an existing notebook (`/nb 00_setup`).
- Update `.agents/STATUS.md` after a change.
- Add a dependency to `pyproject.toml` and `uv sync`.
- Add a cleared source to `config.py::SOURCES`.
Avoid handing them open-ended design (e.g. "design the training loop") cold —
give them the structure first.
