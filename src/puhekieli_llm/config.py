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
SOURCES: dict[str, dict[str, str]] = {
    # examples — none active until the user says so:
    # "opensubtitles_fi": {"register": "puhekieli", "status": "planned",
    #                       "note": "movie/TV subtitles — spoken register, check license"},
    # "suomi24":          {"register": "puhekieli", "status": "planned",
    #                       "note": "forum posts — very colloquial"},
    # "tatoeba":          {"register": "mixed", "status": "planned",
    #                       "note": "EN-FI sentence pairs, mostly kirjakieli"},
}


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
