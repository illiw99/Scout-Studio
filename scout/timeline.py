"""Turning a folder of camera files into one continuous timeline.

A DJI camera splits a single recording at 4 GB, so a game arrives as several
files whose play action runs straight across the joins.  Everything downstream
works in *global* seconds since the start of the first file of a recording, and
this module converts between that clock and (file, local offset).
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .media import MediaInfo, probe
from .storage import is_state_path


def filename_key(path: Path):
    """Camera sequence order, including numbered names and edited suffixes."""
    def natural(text):
        return tuple(int(part) if part.isdigit() else part.casefold()
                     for part in re.split(r'(\d+)', text))
    return natural(path.name), natural(str(path))


@dataclass
class Clip:
    """One source file placed on the global timeline."""
    info: MediaInfo
    start: float          # global seconds
    chapter: int          # index of the recording this file belongs to

    @property
    def end(self) -> float:
        return self.start + self.info.duration

    @property
    def path(self) -> Path:
        return self.info.path


def stable_set(paths: list[Path], seconds: float) -> set[Path]:
    """Which of ``paths`` have stopped growing.

    All files are watched over a single shared wait rather than one wait each,
    so adding more files to the folder does not make startup slower.
    """
    if seconds <= 0:
        return set(paths)
    sizes = {p: p.stat().st_size for p in paths}
    time.sleep(seconds)
    out = set()
    for p in paths:
        try:
            if p.stat().st_size == sizes[p]:
                out.add(p)
        except OSError:
            pass
    return out


def discover(cfg: Config, wait_for_stable: bool = True) -> list[Path]:
    """Ordered source files, excluding application state and files still copying.

    A linked source folder takes precedence over the legacy media tree.
    """
    root = Path(cfg.source_dir if cfg.source_dir is not None else cfg.media_dir)
    if not root.is_dir():
        raise SystemExit(f"no media directory: {root}")
    exts = {e.lower() for e in cfg.extensions}
    files = sorted((p for p in root.rglob("*")
                   if not is_state_path(p.relative_to(root))
                   and p.is_file() and p.suffix.lower() in exts), key=filename_key)
    if not files:
        raise SystemExit(f"no video files in {root}")
    if not wait_for_stable:
        return files

    ok = stable_set(files, cfg.stability_seconds)
    ready = [p for p in files if p in ok]
    for p in files:
        if p not in ok:
            print(f"  skipping {p.name}: still being written")
    if not ready:
        raise SystemExit("every file is still being copied - try again shortly")
    return ready


def build(cfg: Config, paths: list[Path] | None = None) -> list[Clip]:
    """Probe the files, order them and lay them on a shared clock.

    Files follow their numbered filename sequence. Recording timestamps group
    them into chapters: a file that starts within ``chapter_gap_tolerance`` of the
    previous one ending is a continuation, and shares its clock.
    """
    paths = paths or discover(cfg)
    infos = []
    for p in paths:
        try:
            infos.append(probe(p))
        except ValueError as exc:
            # A file that is still being copied has no moov atom yet.
            print(f"  skipping {exc}")
    if not infos:
        raise SystemExit("no readable video files yet - try again shortly")
    infos.sort(key=lambda i: filename_key(i.path))

    clips: list[Clip] = []
    chapter = 0
    for i, info in enumerate(infos):
        if i == 0:
            clips.append(Clip(info, 0.0, chapter))
            continue
        prev = clips[-1]
        gap = None
        if info.created and prev.info.created:
            gap = (info.created - prev.info.created).total_seconds() - prev.info.duration
        if gap is not None and abs(gap) <= cfg.chapter_gap_tolerance:
            # Continuation: keep the clock running, absorbing the tiny gap so
            # global time stays monotonic across the join.
            clips.append(Clip(info, prev.end, chapter))
        else:
            chapter += 1
            clips.append(Clip(info, prev.end, chapter))
    return clips


def rebase(clips: list[Clip]) -> list[Clip]:
    """Shift a selection so its first file starts at zero.

    Times are only meaningful relative to the set being worked on: a project
    holding one file expects that file to begin at 00:00.
    """
    if not clips:
        return clips
    off = clips[0].start
    if off == 0:
        return clips
    return [Clip(c.info, c.start - off, c.chapter) for c in clips]


def locate(clips: list[Clip], t: float) -> tuple[Clip, float]:
    """Map a global time to the clip containing it and the offset within it."""
    for c in clips:
        if c.start <= t < c.end:
            return c, t - c.start
    last = clips[-1]
    return last, min(t - last.start, last.info.duration)


def describe(clips: list[Clip]) -> str:
    lines = []
    for c in clips:
        lines.append(
            f"  [ch{c.chapter}] {c.info.name:>16}  "
            f"{c.start:8.1f}s -> {c.end:8.1f}s  ({c.info.duration/60:5.2f} min)"
        )
    total = clips[-1].end if clips else 0.0
    lines.append(f"  total {total/60:.1f} min across {len(clips)} file(s), "
                 f"{clips[-1].chapter + 1 if clips else 0} recording(s)")
    return "\n".join(lines)
