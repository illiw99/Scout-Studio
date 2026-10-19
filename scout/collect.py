"""Gathering the plays of one game into a single named folder.

A game is recorded across several camera files, and each file has been reviewed
on its own, so its plays are numbered from one. A game is what people actually
want to look through, so this copies every play of a game into one folder,
numbered straight through in playing order, under the name the game is known by.

Clips are copied, never moved: the per-file output stays exactly as it is.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .cut import name_for
from . import groundtruth
from .media import probe


@dataclass
class Collected:
    index: int              # play number within the game
    source_file: str
    source_index: int       # play number within that camera file
    start: float            # seconds into the camera file
    end: float
    game_time: float        # seconds into the whole game
    src: Path
    dst: Path


def _gt_path(out_dir: Path, stem: str) -> Path | None:
    """The reviewed project for a camera file, whatever it ended up called."""
    d = out_dir / stem
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.mlt")):
        if p.stem in ("review", "plays"):
            continue
        if groundtruth.media_for(p):
            return p
    return None


def plan(cfg: Config, game: str, stems: list[str], dest_root: Path,
         media_dir: Path) -> tuple[list[Collected], list[str]]:
    out_dir = cfg.path(cfg.out_dir)
    dest = dest_root / game
    rows: list[Collected] = []
    warnings: list[str] = []
    n = 0
    elapsed = 0.0
    extensions = {e.lower() for e in cfg.extensions}
    sources: dict[str, list[Path]] = {}
    for p in sorted(media_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            sources.setdefault(p.stem, []).append(p)

    for stem in stems:
        gt = _gt_path(out_dir, stem)
        clips_dir = out_dir / stem / "clips"
        matches = sources.get(stem, [])
        if len(matches) > 1:
            raise SystemExit(f"{stem}: multiple media files found: "
                             + ", ".join(str(p) for p in matches))
        duration = 0.0
        try:
            if not matches:
                raise FileNotFoundError(stem)
            duration = probe(matches[0]).duration
        except Exception:
            warnings.append(f"{stem}: cannot read duration; game times after "
                            f"this file will be off")

        if gt is None:
            warnings.append(f"{stem}: no reviewed .mlt found - skipped")
            elapsed += duration
            continue
        if not clips_dir.is_dir():
            warnings.append(f"{stem}: no clips folder - skipped")
            elapsed += duration
            continue

        marks = groundtruth.load(gt)
        plays = groundtruth.to_plays(marks)
        missing = 0
        for m, pl in zip(marks, plays):
            src = clips_dir / name_for(pl)
            if not src.exists():
                missing += 1
                continue
            n += 1
            rows.append(Collected(
                index=n, source_file=stem, source_index=m.index,
                start=m.start, end=m.end, game_time=elapsed + m.start,
                src=src,
                dst=dest / f"{game} - play {n:03d}.mp4",
            ))
        if missing:
            warnings.append(f"{stem}: {missing} clip(s) listed in the .mlt were "
                            f"not on disk")
        elapsed += duration

    return rows, warnings


def run(cfg: Config, game: str, stems: list[str], dest_root: Path,
        media_dir: Path, dry_run: bool = False) -> list[Collected]:
    rows, warnings = plan(cfg, game, stems, dest_root, media_dir)
    for w in warnings:
        print(f"  WARNING: {w}")
    if not rows:
        raise SystemExit(f"{game}: nothing to collect")

    dest = dest_root / game
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows, 1):
            shutil.copy2(r.src, r.dst)
            print(f"\r  copying {i}/{len(rows)}", end="", flush=True)
        print()

        with (dest / "plays.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["play", "clip", "game_time_s", "game_time",
                        "source_file", "source_play", "start_s", "end_s",
                        "duration_s"])
            for r in rows:
                m, s = divmod(int(r.game_time), 60)
                h, m = divmod(m, 60)
                w.writerow([r.index, r.dst.name, f"{r.game_time:.2f}",
                            f"{h:d}:{m:02d}:{s:02d}", r.source_file,
                            r.source_index, f"{r.start:.2f}", f"{r.end:.2f}",
                            f"{r.end - r.start:.2f}"])
    return rows


def rename(game: str, prefix: str, root: Path, pad: int = 3) -> int:
    """Rename a game's clips to ``<prefix>_<number>.mp4`` and update the table.

    Numbers stay zero-padded so the files sort in playing order in any file
    browser, which they would not do past play 9 otherwise.
    """
    folder = root / game
    csv_path = folder / "plays.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found")
    rows = _rows_of(csv_path)

    pending = []
    for r in rows:
        old = folder / r["clip"]
        new_name = f"{prefix}_{int(r['play']):0{pad}d}{old.suffix}"
        pending.append((old, folder / new_name, r, new_name))

    missing = [str(o.name) for o, _, _, _ in pending if not o.exists()]
    if missing:
        raise SystemExit(f"{game}: missing {len(missing)} clip(s), e.g. {missing[0]}")

    for old, new, r, new_name in pending:
        if old != new:
            old.rename(new)
        r["clip"] = new_name

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _rows_of(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))
