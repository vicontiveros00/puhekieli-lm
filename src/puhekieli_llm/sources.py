"""Data collection helpers for puhekieli-llm.

Sources feeding the pipeline (see config.SOURCES):
  - genius_rap        : Finnish rap lyrics (FI-only "flavor"). Genius fully blocks
                        lyric scraping behind Cloudflare Private Access Tokens, so
                        we do NOT scrape lyrics. Instead:
                          * `fetch_genius_metadata()` uses the working Genius API to
                            list an artist's songs + URLs (no lyric text).
                          * you paste lyric text you like into
                            data/raw/genius_rap/*.txt (a CURATED seed), and
                            `clean_genius_lyrics()` turns it into records.
  - opensubtitles_enfi: EN-FI parallel "pairs", via HF (streamed, no giant zip).
  - opus_100          : EN-FI parallel "pairs", broader/varied (HF, streamed).

Cleaned schema:
  flavor (FI-only):       {"source","id","fi","register","meta":{...}}
  parallel pairs (EN-FI): {"source","id","en","fi","register"}
Keep raw around so we can re-clean without re-fetching.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

from puhekieli_llm.config import CLEAN, RAW, RAP_ARTISTS


# --- shared jsonl io ------------------------------------------------------
def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- text cleaning --------------------------------------------------------
_BRACKET_TAG = re.compile(r"^\s*[\[\(].*?[\]\)]\s*$")  # [Kertosäe], (x2), [Verse 1]
_GENIUS_CRUFT = re.compile(r"\d*Embed$|You might also like|Lyrics$", re.IGNORECASE)


def clean_lyric_lines(raw: str) -> list[str]:
    """Split raw Genius lyric text into clean puhekieli lines.

    Drops section headers ([Kertosäe]), Genius UI cruft, empty lines, and
    obvious English-only lines are KEPT (Finnish rap code-switches a lot, and
    that mixing is authentic puhekieli-in-the-wild).
    """
    out: list[str] = []
    for line in raw.splitlines():
        line = _GENIUS_CRUFT.sub("", line).strip()
        if not line or _BRACKET_TAG.match(line):
            continue
        # collapse whitespace
        line = re.sub(r"\s+", " ", line)
        out.append(line)
    return out


def dedup(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return out


# --- Genius rap lyrics (flavor, FI-only) ----------------------------------
def fetch_genius_metadata(
    artists: list[str] | None = None,
    max_songs_per_artist: int = 40,
    token: str | None = None,
) -> Path:
    """List songs per artist via the Genius API (metadata only, NO lyric text).

    Genius blocks lyric-page scraping behind Cloudflare Private Access Tokens
    (hardware attestation) which automation cannot pass. The JSON API still works
    though, so we use it to build a per-artist song list (title + Genius URL) into
    data/raw/genius_rap/<artist>.songs.json. Use it to pick which songs to seed:
    open the URL in your own browser and paste lyrics into <artist>.txt.

    Requires a Genius token (https://genius.com/api-clients); pass it or set
    GENIUS_ACCESS_TOKEN (config.py loads .env).
    """
    import httpx

    artists = artists or RAP_ARTISTS
    token = token or os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "No Genius token. Get one at https://genius.com/api-clients and put it "
            "in .env as GENIUS_ACCESS_TOKEN."
        )
    headers = {"Authorization": f"Bearer {token}"}
    raw_dir = RAW / "genius_rap"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in artists:
        # search returns hits across artists; keep only this artist's songs
        songs: list[dict] = []
        page = 1
        while len(songs) < max_songs_per_artist and page <= 5:
            r = httpx.get(
                "https://api.genius.com/search",
                params={"q": name, "per_page": 20, "page": page},
                headers=headers, timeout=20,
            )
            r.raise_for_status()
            hits = r.json()["response"]["hits"]
            if not hits:
                break
            for h in hits:
                s = h["result"]
                if s["primary_artist"]["name"].lower() == name.lower():
                    songs.append({"title": s["title"], "url": s["url"], "id": s["id"]})
            page += 1
        songs = songs[:max_songs_per_artist]
        out = raw_dir / f"{name.lower().replace(' ', '_')}.songs.json"
        out.write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}: {len(songs)} songs listed -> {out.name}")
    print("\nNext: open the URLs, paste lyrics you like into "
          f"{raw_dir}/<artist>.txt, then run clean_genius_lyrics().")
    return raw_dir


def clean_genius_lyrics() -> Path:
    """Clean curated seed lyrics -> data/clean/genius_rap.jsonl (one line per record).

    Reads every data/raw/genius_rap/*.txt you pasted lyrics into. Format is plain
    text: one lyric line per line. Blank lines are skipped; lines starting with
    '#' are treated as comments (e.g. '# Gettomasa - Lössi' to mark a song) and
    used as the current title tag. FI-only records; the FI text is the authentic
    rap-register puhekieli line.
    """
    raw_dir = RAW / "genius_rap"
    all_lines: list[tuple[str, str, str]] = []  # (artist, title, line)
    for tf in sorted(raw_dir.glob("*.txt")):
        if tf.stem.lower() in {"readme", "_readme", "example"}:
            continue
        artist = tf.stem.replace("_", " ")
        title = ""
        for raw_line in tf.read_text(encoding="utf-8").splitlines():
            s = raw_line.strip()
            if not s:
                continue
            if s.startswith("#"):
                title = s.lstrip("#").strip()
                continue
            for ln in clean_lyric_lines(s):
                all_lines.append((artist, title, ln))

    if not all_lines:
        print(f"No seed lyrics found. Paste lines into {raw_dir}/<artist>.txt "
              "(see fetch_genius_metadata for song URLs), then re-run.")
        return CLEAN / "genius_rap.jsonl"

    # dedup on the line text (choruses repeat a lot)
    seen: set[str] = set()
    records: list[dict] = []
    idx = 0
    for artist, title, line in all_lines:
        key = line.lower()
        if key in seen or len(line) < 8:  # drop trivially short lines
            continue
        seen.add(key)
        records.append({
            "source": "genius_rap",
            "id": f"genius_rap-{idx}",
            "fi": line,
            "register": "puhekieli",
            "meta": {"artist": artist, "title": title},
        })
        idx += 1

    out = CLEAN / "genius_rap.jsonl"
    n = write_jsonl(out, records)
    print(f"genius_rap: {n} curated lyric lines -> {out}")
    return out


# --- EN-FI parallel pairs (via HuggingFace, streamed) ----------------------
# We stream from HF so we never materialize the huge corpora or need the OPUS
# moses zip. Both loaders below are plain data (parquet/text), not scripts, so
# they work with modern `datasets`.
def _clean_pair(source: str, i: int, en: str, fi: str, register: str) -> dict | None:
    en, fi = en.strip(), fi.strip()
    if not en or not fi:
        return None
    if not (2 <= len(en) <= 200 and 2 <= len(fi) <= 200):
        return None
    return {
        "source": source,
        "id": f"{source}-{i}",
        "en": re.sub(r"\s+", " ", en),
        "fi": re.sub(r"\s+", " ", fi),
        "register": register,
    }


def fetch_opensubtitles(max_pairs: int = 200_000) -> Path:
    """Stream OpenSubtitles EN-FI from HF -> data/clean/opensubtitles_enfi.jsonl.

    Uses `sentence-transformers/parallel-sentences-opensubtitles` (en-fi), which
    is deduped subtitle dialogue — colloquial-leaning. `max_pairs` caps it.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "sentence-transformers/parallel-sentences-opensubtitles",
        "en-fi", split="train", streaming=True,
    )

    def _records() -> Iterator[dict]:
        kept = 0
        for i, ex in enumerate(ds):
            rec = _clean_pair("opensubtitles_enfi", i, ex["english"], ex["non_english"], "mixed")
            if rec is None:
                continue
            yield rec
            kept += 1
            if kept >= max_pairs:
                break

    out = CLEAN / "opensubtitles_enfi.jsonl"
    n = write_jsonl(out, _records())
    print(f"opensubtitles_enfi: {n} EN-FI pairs -> {out}")
    return out


def fetch_opus100(max_pairs: int = 200_000) -> Path:
    """Stream OPUS-100 EN-FI from HF -> data/clean/opus_100.jsonl.

    `Helsinki-NLP/opus-100` en-fi is a broad, mixed-domain parallel corpus (~1M
    pairs) — good general translation signal to complement the subtitle dialogue.
    """
    from datasets import load_dataset

    ds = load_dataset("Helsinki-NLP/opus-100", "en-fi", split="train", streaming=True)

    def _records() -> Iterator[dict]:
        kept = 0
        for i, ex in enumerate(ds):
            t = ex["translation"]
            rec = _clean_pair("opus_100", i, t["en"], t["fi"], "mixed")
            if rec is None:
                continue
            yield rec
            kept += 1
            if kept >= max_pairs:
                break

    out = CLEAN / "opus_100.jsonl"
    n = write_jsonl(out, _records())
    print(f"opus_100: {n} EN-FI pairs -> {out}")
    return out
