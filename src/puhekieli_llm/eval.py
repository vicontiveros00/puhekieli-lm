"""Puhekieli-feature scoring.

BLEU against a kirjakieli reference would PENALIZE the spoken forms we want, so we
also score how "spoken" a Finnish string looks. This is a cheap heuristic, not
linguistics — it counts markers of colloquial/Helsinki speech vs. their formal
written equivalents. Higher = more puhekieli.
"""
from __future__ import annotations

import re

# spoken pronouns / forms (puhekieli) — presence is good
_PUHE_MARKERS = [
    r"\bmä\b", r"\bmää\b", r"\bmun\b", r"\bmul\b", r"\bmua\b",     # minä-forms
    r"\bsä\b", r"\bsää\b", r"\bsun\b", r"\bsul\b", r"\bsua\b",     # sinä-forms
    r"\bne\b", r"\bniiden\b", r"\bniil\b",                          # he -> ne
    r"\bmennään\b", r"\btehään\b", r"\bollaan\b",                   # me-passive
    r"\btää\b", r"\btän\b", r"\btää\b", r"\bnää\b",                 # tämä/nämä
    r"\bemmä\b", r"\bemmä\b", r"\btiiä\b", r"\bnääks\b", r"\bmeen\b",
    r"\bstadi\b", r"\bbroidi\b", r"\bfrendi\b", r"\bskeidaa\b",     # stadin slangi
]
# formal written forms (kirjakieli) — presence is "bad" for our goal
_KIRJA_MARKERS = [
    r"\bminä\b", r"\bsinä\b", r"\bhän\b", r"\bhe\b",
    r"\bolen\b", r"\bolemme\b", r"\bmenemme\b", r"\bteemme\b",
    r"\btämä\b", r"\bnämä\b", r"\bemme\b",
]

_PUHE = [re.compile(p, re.IGNORECASE) for p in _PUHE_MARKERS]
_KIRJA = [re.compile(p, re.IGNORECASE) for p in _KIRJA_MARKERS]


def puhekieli_score(text: str) -> dict:
    """Return spoken/formal marker counts and a normalized score in [-1, 1].

    +1 = clearly spoken, -1 = clearly formal written, 0 = neither/ambiguous.
    """
    puhe = sum(len(rx.findall(text)) for rx in _PUHE)
    kirja = sum(len(rx.findall(text)) for rx in _KIRJA)
    total = puhe + kirja
    score = 0.0 if total == 0 else (puhe - kirja) / total
    return {"puhe": puhe, "kirja": kirja, "score": round(score, 3)}


def corpus_puhekieli_rate(texts: list[str]) -> float:
    """Fraction of texts that lean spoken (score > 0)."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if puhekieli_score(t)["score"] > 0) / len(texts)
