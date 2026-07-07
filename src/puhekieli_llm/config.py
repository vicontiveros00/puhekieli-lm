"""Shared config + paths for the puhekieli-llm project.

Single source of truth so every notebook/script agrees on where data lives
and which device to use. Import this everywhere instead of hardcoding paths.
"""
from __future__ import annotations

from pathlib import Path

import torch

# --- Project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"           # untouched collected source data
CLEAN = DATA / "clean"       # normalized, deduped parallel/text (jsonl)
TOKENIZED = DATA / "tokenized"  # token id arrays (.bin) ready for training
MODELS = ROOT / "models"     # tokenizer + checkpoints
CONFIGS = ROOT / "configs"

for _p in (RAW, CLEAN, TOKENIZED, MODELS, CONFIGS):
    _p.mkdir(parents=True, exist_ok=True)


# --- Language / task config ----------------------------------------------
SRC_LANG = "en"   # source: English
TGT_LANG = "fi"   # target: Finnish
# The whole point: target the SPOKEN register, not the written standard.
TGT_REGISTER = "puhekieli"   # spoken/colloquial Finnish (vs. "kirjakieli")


# --- Data source registry -------------------------------------------------
# The USER chooses the sources. This is a pluggable registry so a source can be
# added without reworking the pipeline. Fill in as you go. For each source track
# its Finnish register (we want spoken/colloquial) and a `status`:
#   "planned"  -> considering it, not collected yet
#   "active"   -> cleared by user, being used
#   "excluded" -> decided against (log why in DECISIONS.md)
# `register`: "puhekieli" (spoken), "mixed", or "kirjakieli" (written; usually
# not what we want, but useful as contrast / for the source side).
# `role`: how the source is used in the pipeline:
#   "pairs"    -> real EN->FI parallel training pairs
#   "flavor"   -> FI-only text; drives tokenizer + puhekieli eval
#   "synth"    -> synthetic EN->FI pairs derived by back-translation (FI is real)
SOURCES: dict[str, dict[str, str]] = {
    "opensubtitles_enfi": {
        "register": "mixed", "role": "pairs", "status": "active",
        "note": "OPUS OpenSubtitles EN-FI — dialogue, leans colloquial; the base pairs",
    },
    "genius_rap": {
        "register": "puhekieli", "role": "flavor", "status": "active",
        "note": "Finnish rap lyrics via Genius API — pure Helsinki puhekieli/slang, FI-only",
    },
    "rap_synthetic": {
        "register": "puhekieli", "role": "synth", "status": "active",
        "note": "EN->FI pairs synthesized from genius_rap by local-LLM back-translation; FI side is the real lyric",
    },
}

# Rap artists to seed the Genius scrape (young Helsinki spoken/slang register).
RAP_ARTISTS: list[str] = ["Gettomasa", "JVG", "Ibe", "Etta", "Costi"]


# --- Local LLM (LM Studio) for back-translation ---------------------------
# Synthetic-pair generation calls a local OpenAI-compatible endpoint. Nothing
# leaves the machine. Override via env if your setup differs.
import os

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")  # whatever is loaded in LM Studio
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")  # LM Studio ignores the value


def active_sources() -> list[str]:
    """Sources the user has explicitly cleared for use."""
    return [name for name, meta in SOURCES.items() if meta.get("status") == "active"]


def get_device() -> torch.device:
    """Prefer Apple Metal (MPS), fall back to CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def summary() -> str:
    active = active_sources() or ["(none yet — user picks sources)"]
    return (
        f"task      : {SRC_LANG} -> {TGT_LANG} ({TGT_REGISTER})\n"
        f"root      : {ROOT}\n"
        f"device    : {DEVICE}\n"
        f"raw       : {RAW}\n"
        f"clean     : {CLEAN}\n"
        f"tokenized : {TOKENIZED}\n"
        f"models    : {MODELS}\n"
        f"sources   : {', '.join(active)}"
    )
