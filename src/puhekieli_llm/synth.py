"""Synthetic EN->FI pair generation via a local LLM (LM Studio).

The idea (back-translation): rap lyrics are FI-only, and OpenSubtitles Finnish is
only mildly colloquial. To teach the model to PRODUCE hard rap-register puhekieli
from English, we manufacture the missing English side:

    real FI lyric  --(local LLM: FI->EN)-->  synthetic EN
    training pair  =  (synthetic EN  ->  real FI lyric)

The Finnish side is ALWAYS the authentic lyric, so the *target* the model learns
to generate is genuine puhekieli. Only the English (input) side is synthetic, and
input noise matters far less than output noise.

Everything runs locally against LM Studio's OpenAI-compatible endpoint — nothing
leaves the machine.
"""
from __future__ import annotations

import time
from pathlib import Path

import re
import time
from typing import Iterator

from puhekieli_llm.config import (
    CLEAN,
    LMSTUDIO_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
)
from puhekieli_llm.sources import read_jsonl, write_jsonl


_SYSTEM = (
    "You are a translator. You are given a line of colloquial/slang Finnish "
    "(spoken register, often Helsinki rap lyrics). Translate it into natural, "
    "everyday English. Output ONLY the English translation, no quotes, no notes. "
    "If the line is already English or nonsense, repeat it unchanged."
)

# Both models we tested (qwen3-14b, gpt-oss-20b) are reasoning models. They must
# be allowed to finish thinking or `content` comes back empty:
#   - qwen3: honors `/no_think` to skip reasoning entirely (fast).
#   - gpt-oss: ignores /no_think but reasons briefly, THEN answers — it just needs
#     a big enough token budget to reach the answer (else finish_reason=length).
# So we keep the prompt clean and give a generous max_tokens. `/no_think` is
# appended automatically for qwen-style models (harmless to others via strip).
_MAX_TOKENS = 1024  # room for reasoning models to think AND answer


def _strip_output(text: str) -> str:
    """Tidy model output: drop wrapping quotes and stray reasoning tags."""
    text = text.strip()
    # some reasoning models leak a <think>...</think> block into content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # strip a single pair of surrounding quotes the model likes to add
    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    return text


def _client():
    from openai import OpenAI

    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def back_translate_line(fi: str, client=None, model: str | None = None,
                        temperature: float = 0.2) -> str:
    """FI -> EN for a single line via the local LLM.

    Reasoning models (qwen3, gpt-oss) must be allowed to finish thinking or
    `content` comes back empty. We give a generous token budget (_MAX_TOKENS) and,
    for qwen-style models, append `/no_think` to skip reasoning entirely for speed.
    Non-reasoning models are unaffected.
    """
    client = client or _client()
    model = model or LMSTUDIO_MODEL
    system = _SYSTEM
    if "qwen" in model.lower():
        system = system + " /no_think"
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
        extra_body={"reasoning_effort": "low"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": fi},
        ],
    )
    return _strip_output(resp.choices[0].message.content or "")


def synthesize_pairs(
    src_jsonl: Path | None = None,
    limit: int | None = 2000,
    model: str | None = None,
) -> Path:
    """Read FI-only lyric lines, back-translate to EN, write EN->FI pairs.

    src_jsonl defaults to data/clean/genius_rap.jsonl. Output:
    data/clean/rap_synthetic.jsonl with {source,id,en,fi,register}. Resumable:
    skips ids already present in the output file.
    """
    src = src_jsonl or (CLEAN / "genius_rap.jsonl")
    out = CLEAN / "rap_synthetic.jsonl"

    done_ids: set[str] = set()
    existing: list[dict] = []
    if out.exists():
        existing = list(read_jsonl(out))
        done_ids = {r["id"] for r in existing}
        print(f"resuming: {len(done_ids)} pairs already synthesized")

    client = _client()
    rows = list(read_jsonl(src))
    if limit:
        rows = rows[:limit]

    new: list[dict] = []
    for i, r in enumerate(rows):
        pid = f"synth-{r['id']}"
        if pid in done_ids:
            continue
        fi = r["fi"]
        try:
            en = back_translate_line(fi, client=client, model=model)
        except Exception as exc:  # noqa: BLE001 — demo: skip a bad line, keep going
            print(f"  ! {r['id']} failed: {exc}")
            continue
        if not en or len(en) < 2:
            continue
        new.append({
            "source": "rap_synthetic",
            "id": pid,
            "en": en,
            "fi": fi,               # REAL puhekieli target
            "register": "puhekieli",
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} back-translated")
            # flush incrementally so a crash doesn't lose progress
            write_jsonl(out, existing + new)

    write_jsonl(out, existing + new)
    print(f"rap_synthetic: {len(existing) + len(new)} total pairs -> {out}")
    return out
