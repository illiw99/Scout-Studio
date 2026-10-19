"""Writing each play out as its own video file.

Two subtleties.  A play can straddle the 4 GB point where the camera split the
recording, so such a clip is rendered in parts and joined.  And plain stream
copying can only start at a keyframe, which on this footage means up to a second
of drift, so the default is to re-encode: slower, but the cut lands where the
play actually starts.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import tempfile
from pathlib import Path

from .config import Config
from .media import ffmpeg_exe, run
from .segment import Play
from .timeline import Clip


def _encode_args(cfg: Config) -> list[str]:
    if cfg.cut_mode == "copy":
        return ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    return [
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
    ]


def _covering(clips: list[Clip], start: float, end: float) -> list[Clip]:
    return [c for c in clips if c.end > start and c.start < end]


def _cut_part(src: Path, local_start: float, dur: float, dst: Path,
              cfg: Config) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{local_start:.3f}", "-i", str(src), "-t", f"{dur:.3f}"]
    cmd += _encode_args(cfg)
    cmd += ["-movflags", "+faststart", str(dst)]
    run(cmd)


def cut_one(cfg: Config, clips: list[Clip], play: Play, dst: Path) -> Path:
    parts = _covering(clips, play.start, play.end)
    if play.start < 0 or play.end <= play.start or play.end > clips[-1].end + .002:
        raise ValueError(f"play {play.index} has invalid bounds")
    if not parts:
        raise ValueError(f"play {play.index} falls outside the timeline")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if len(parts) == 1:
        c = parts[0]
        _cut_part(c.path, play.start - c.start,
                  min(play.end, c.end) - play.start, dst, cfg)
        return dst

    # Spans a camera file split: render each side, then join them.
    tmp = Path(tempfile.mkdtemp(prefix="scout_"))
    try:
        segs = []
        for i, c in enumerate(parts):
            s = max(play.start, c.start)
            e = min(play.end, c.end)
            if e - s < 0.05:
                continue
            part = tmp / f"part{i}.mp4"
            _cut_part(c.path, s - c.start, e - s, part, cfg)
            segs.append(part)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in segs))
        run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(dst)])
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def name_for(play: Play) -> str:
    """e.g. play_007_06m11s.mp4, or play_050_02h15m30s.mp4 past the hour."""
    mm, ss = divmod(int(play.snap), 60)
    hh, mm = divmod(mm, 60)
    stamp = f"{hh:02d}h{mm:02d}m{ss:02d}s" if hh else f"{mm:02d}m{ss:02d}s"
    return f"play_{play.index:03d}_{stamp}.mp4"


def cut_all(cfg: Config, clips: list[Clip], plays: list[Play],
            out_dir: Path | None = None, progress: bool = True, partial: bool = False) -> list[Path]:
    out_dir = out_dir or cfg.path(cfg.out_dir) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(prefix=".export-", dir=out_dir)
    stage = Path(staging.name)
    done: list[Path] = []
    total = len(plays)
    failures = []

    def job(p: Play) -> Path:
        return cut_one(cfg, clips, p, stage / name_for(p))

    with cf.ThreadPoolExecutor(max_workers=cfg.clip_workers) as ex:
        futs = {ex.submit(job, p): p for p in plays}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            p = futs[fut]
            try:
                done.append(fut.result())
            except Exception as exc:
                failures.append(f"play {p.index}: {exc}")
            if progress:
                print(f"\r  cutting {i}/{total} clips", end="", flush=True)
    if progress:
        print()
    if failures:
        staging.cleanup()
        raise RuntimeError("clip export failed: " + "; ".join(failures))
    manifest = out_dir / "exports.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    current = [p.name for p in done]
    if partial:
        current = sorted(set(previous) | set(current))
    for path in done:
        path.replace(out_dir / path.name)
    for name in set(previous) - set(current):
        if Path(name).name == name and name.startswith("play_") and name.endswith(".mp4"):
            (out_dir / name).unlink(missing_ok=True)
    temporary = out_dir / ".exports.json.tmp"
    temporary.write_text(json.dumps(current, indent=2))
    temporary.replace(manifest)
    staging.cleanup()
    return sorted(out_dir / name for name in current)
