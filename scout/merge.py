"""Joining plays that the camera split in half.

The camera starts a new file every 4 GB, and a play running across that moment
is reviewed as two markers in two different projects, so it arrives as two
clips. Joining them restores the play. The pieces are stream-copied, since they
came from the same camera with the same settings, so nothing is re-encoded.

Everything after the join is renumbered to close the gap.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

from .media import ffmpeg_exe, run


def _rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def _clip_name(game: str, n: int) -> str:
    return f"{game} - play {n:03d}.mp4"


def merge(game: str, plays: list[int], root: Path) -> tuple[int, int]:
    """Join consecutive plays into the first of them; renumber the rest."""
    plays = sorted(set(plays))
    if len(plays) < 2:
        raise SystemExit("give at least two play numbers to join")
    if plays != list(range(plays[0], plays[-1] + 1)):
        raise SystemExit(f"plays must be consecutive, got {plays}")

    folder = root / game
    csv_path = folder / "plays.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found")
    rows = _rows(csv_path)
    by_num = {int(r["play"]): r for r in rows}
    for n in plays:
        if n not in by_num:
            raise SystemExit(f"{game}: no play {n}")
        if not (folder / by_num[n]["clip"]).exists():
            raise SystemExit(f"missing file: {by_num[n]['clip']}")

    keep, drop = plays[0], plays[1:]
    tmp = Path(tempfile.mkdtemp(prefix="scout_merge_"))
    try:
        listing = tmp / "list.txt"
        listing.write_text("".join(
            f"file '{(folder / by_num[n]['clip']).resolve()}'\n" for n in plays))
        joined = tmp / "joined.mp4"
        run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(joined)])

        for n in plays:
            (folder / by_num[n]["clip"]).unlink()
        shutil.move(str(joined), folder / _clip_name(game, keep))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Rebuild the table: the joined play spans its pieces, and everything after
    # it shifts down to close the gap.
    merged_row = dict(by_num[keep])
    parts = [by_num[n] for n in plays]
    merged_row["duration_s"] = f"{sum(float(p['duration_s']) for p in parts):.2f}"
    merged_row["source_file"] = "+".join(dict.fromkeys(p["source_file"] for p in parts))
    merged_row["source_play"] = "+".join(p["source_play"] for p in parts)
    merged_row["end_s"] = parts[-1]["end_s"]

    out: list[dict] = []
    n = 0
    for r in rows:
        num = int(r["play"])
        if num in drop:
            continue
        n += 1
        r = merged_row if num == keep else dict(r)
        old = r["clip"]
        r["play"] = str(n)
        r["clip"] = _clip_name(game, n)
        if num != keep and old != r["clip"]:
            src, dst = folder / old, folder / r["clip"]
            if src.exists():
                src.rename(dst)
        out.append(r)

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out)
    return len(rows), len(out)
