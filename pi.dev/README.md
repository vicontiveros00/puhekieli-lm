# pi.dev — running this repo with weaker local models

This repo (like its solita-llm parent) is set up so weaker local models (go-to:
**Qwen3 14B**, **gpt-oss-20b**) stay on the rails. Local models are capable but
drift more than frontier models. This folder explains the strategy; the actual
machinery lives in standard pi locations so pi loads it automatically.

## What's wired up (and where pi loads it from)

| Thing | Location | Purpose |
|-------|----------|---------|
| Context file | `AGENTS.md` (repo root) | Short, imperative rules pi auto-loads at startup |
| Handoff docs | `.agents/*.md` | Project/status/decisions/env/data-policy |
| Skill | `.agents/skills/repo-tasks/` | On-demand step-by-step playbooks (`/skill:repo-tasks`) |
| Model notes | `.agents/skills/repo-tasks/references/MODEL_NOTES.md` | Small-model gotchas + tips |
| Prompt templates | `.pi/prompts/*.md` | Canned exact commands: `/status`, `/nb`, `/commit` |

Why this layout: weak models do best with **short context + explicit procedures +
canned commands**. AGENTS.md stays tight (rules only); the long procedures live in
a skill that loads on demand (progressive disclosure), so the base context stays
small and cheap.

> **Local-model provider setup is intentionally not documented here yet.** Local
> inference is via LM Studio, but no current task requires a documented pi provider
> setup. When there's a real need, add it then (pi docs: `docs/models.md`,
> `docs/custom-provider.md`).

## Prompt templates (type these in pi)
- `/status` — read STATUS.md + git state, report the next action in 5 lines.
- `/nb <name>` — run a notebook headless to verify, then strip outputs.
- `/commit <message>` — safe commit (forces correct cwd, checks no data/ staged).

## How to drive a weak model here (token-efficient)
1. Start with `/status` (loads only what's needed).
2. Give ONE task at a time. Ask for a 1-line plan, then let it execute.
3. Use `/nb` and `/commit` instead of free-form instructions.
4. If it ignores the skill, force it: `/skill:repo-tasks`.
5. Always make it verify notebooks headless before believing "it works".
6. Remind it the target is **puhekieli**, not formal Finnish — it will drift there.

See `.agents/skills/repo-tasks/references/MODEL_NOTES.md` for the full gotcha list.
