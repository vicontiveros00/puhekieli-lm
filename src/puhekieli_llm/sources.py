"""Data collection helpers for puhekieli-llm.

Two source families feed the pipeline (see config.SOURCES):
  - genius_rap        : Finnish rap lyrics (FI-only "flavor"), via Genius API
  - opensubtitles_enfi: EN-FI parallel "pairs", via OPUS OpenSubtitles

Both write untouched fetches to data/raw/<source>/ and cleaned records to
data/clean/<source>.jsonl. Cleaned schema:
  flavor pairs (FI-only):   {"source","id","fi","register","meta":{...}}
  parallel pairs (EN-FI):   {"source","id","en","fi","register"}
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
def fetch_genius_lyrics(
    artists: list[str] | None = None,
    max_songs_per_artist: int = 40,
    token: str | None = None,
) -> Path:
    """Fetch lyrics for the given artists into data/raw/genius_rap/<artist>.json.

    Requires a Genius API token (https://genius.com/api-clients). Pass it or set
    GENIUS_ACCESS_TOKEN. Returns the raw dir. Network + rate-limited: run once,
    then clean offline.
    """
    import lyricsgenius

    artists = artists or RAP_ARTISTS
    token = token or os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "No Genius token. Get one at https://genius.com/api-clients and set "
            "GENIUS_ACCESS_TOKEN (or pass token=...)."
        )

    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        remove_section_headers=False,  # we strip them ourselves in cleaning
        verbose=True,
        timeout=15,
        retries=3,
    )
    raw_dir = RAW / "genius_rap"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in artists:
        artist = genius.search_artist(name, max_songs=max_songs_per_artist, sort="popularity")
        if artist is None:
            print(f"  ! no results for {name}")
            continue
        payload = [{"title": s.title, "artist": name, "lyrics": s.lyrics} for s in artist.songs]
        out = raw_dir / f"{name.lower().replace(' ', '_')}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}: {len(payload)} songs -> {out.name}")
    return raw_dir


def clean_genius_lyrics() -> Path:
    """Clean raw Genius json -> data/clean/genius_rap.jsonl (one record per line).

    FI-only records; the FI text is the authentic rap-register puhekieli line.
    """
    raw_dir = RAW / "genius_rap"
    records: list[dict] = []
    all_lines: list[tuple[str, str, str]] = []  # (artist, title, line)
    for jf in sorted(raw_dir.glob("*.json")):
        for song in json.loads(jf.read_text(encoding="utf-8")):
            for ln in clean_lyric_lines(song.get("lyrics") or ""):
                all_lines.append((song["artist"], song["title"], ln))

    # dedup on the line text (choruses repeat a lot)
    seen: set[str] = set()
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
    print(f"genius_rap: {n} clean lyric lines -> {out}")
    return out


# --- OpenSubtitles EN-FI (parallel pairs) ---------------------------------
# OPUS distributes moses-format bitext as a single zip containing two aligned
# files (one per language), line-for-line. We download it once into data/raw/
# (gitignored) and stream-clean into a capped jsonl. The HF `open_subtitles`
# loader is script-based and no longer supported by modern `datasets`, so we
# fetch straight from OPUS instead.
OPUS_OPENSUBS_URL = (
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fi.txt.zip"
)


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download url -> dest with a simple progress print. Skips if already there."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"  downloading {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with dest.open("wb") as f:
            for b in r.iter_bytes(chunk):
                f.write(b)
                got += len(b)
                if total:
                    print(f"\r  {got / 1e6:6.0f} / {total / 1e6:.0f} MB", end="")
        print()
    return dest


def fetch_opensubtitles(max_pairs: int = 200_000, keep_zip: bool = True) -> Path:
    """Download OPUS OpenSubtitles EN-FI and stream-clean ->
    data/clean/opensubtitles_enfi.jsonl.

    Reads the two aligned members straight from the zip without extracting them
    to disk, so we only pay the zip's size. `max_pairs` caps the corpus for a
    laptop-friendly demo. Records are lightly length-filtered.
    """
    import zipfile

    raw_dir = RAW / "opensubtitles_enfi"
    zip_path = _download(OPUS_OPENSUBS_URL, raw_dir / "en-fi.txt.zip")

    with zipfile.ZipFile(zip_path) as z:
        en_name = next(n for n in z.namelist() if n.endswith(".en"))
        fi_name = next(n for n in z.namelist() if n.endswith(".fi"))

        def _records() -> Iterator[dict]:
            kept = 0
            with z.open(en_name) as ef, z.open(fi_name) as ff:
                for i, (eb, fb) in enumerate(zip(ef, ff)):
                    en = eb.decode("utf-8", "ignore").strip()
                    fi = fb.decode("utf-8", "ignore").strip()
                    if not en or not fi:
                        continue
                    if not (2 <= len(en) <= 200 and 2 <= len(fi) <= 200):
                        continue
                    yield {
                        "source": "opensubtitles_enfi",
                        "id": f"opensub-{i}",
                        "en": re.sub(r"\s+", " ", en),
                        "fi": re.sub(r"\s+", " ", fi),
                        "register": "mixed",
                    }
                    kept += 1
                    if kept >= max_pairs:
                        break

        out = CLEAN / "opensubtitles_enfi.jsonl"
        n = write_jsonl(out, _records())

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"opensubtitles_enfi: {n} EN-FI pairs -> {out}")
    return out
