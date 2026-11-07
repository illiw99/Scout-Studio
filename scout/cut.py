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
import math
import shutil
import tempfile
from pathlib import Path

from .clip_names import pattern_names
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


def _padding(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite, nonnegative seconds")
    return float(value)


def export_bounds(start: float, end: float, duration: float,
                  pre_play: float, post_play: float) -> tuple[float, float]:
    """Expand a valid core interval for export, clamped to the match timeline.

    The 2 ms end tolerance accommodates review timestamp rounding. Padding
    cannot turn an invalid core interval into a valid export.
    """
    pre_play = _padding(pre_play, "pre-play")
    post_play = _padding(post_play, "post-play")
    if (not all(math.isfinite(value) for value in (start, end, duration))
            or duration <= 0 or start < 0 or start >= duration
            or end <= start or end > duration + .002):
        raise ValueError("play has invalid bounds")
    return max(0., start - pre_play), min(duration, end + post_play)


def _cut_part(src: Path, local_start: float, dur: float, dst: Path,
              cfg: Config, *, concat_part: bool = False) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{local_start:.3f}", "-i", str(src), "-t", f"{dur:.3f}"]
    cmd += _encode_args(cfg)
    if concat_part and cfg.cut_mode != "copy":
        # A one-frame part has no B-frame delay. Use the same zero-delay
        # timestamps for every part so concat cannot overlap or shorten them.
        cmd += ["-bf", "0"]
    cmd += ["-movflags", "+faststart", str(dst)]
    run(cmd)


def cut_one(cfg: Config, clips: list[Clip], play: Play, dst: Path,
            pre_play: float | None = None, post_play: float | None = None) -> Path:
    start, end = export_bounds(play.start, play.end, clips[-1].end if clips else 0.,
                               cfg.cut_pre_play if pre_play is None else pre_play,
                               cfg.cut_post_play if post_play is None else post_play)
    parts = _covering(clips, start, end)
    if not parts:
        raise ValueError(f"play {play.index} falls outside the timeline")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if len(parts) == 1:
        c = parts[0]
        _cut_part(c.path, start - c.start,
                  min(end, c.end) - start, dst, cfg)
        return dst

    # Spans a camera file split: render each side, then join them.
    tmp = Path(tempfile.mkdtemp(prefix="scout_"))
    try:
        segs = []
        for i, c in enumerate(parts):
            s = max(start, c.start)
            e = min(end, c.end)
            if e <= s:
                continue
            part = tmp / f"part{i}.mp4"
            _cut_part(c.path, s - c.start, e - s, part, cfg, concat_part=True)
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
            out_dir: Path | None = None, progress: bool = True, partial: bool = False,
            name_pattern: str | None = None, start_number: int = 1,
            number_width: int | None = None, pre_play: float | None = None,
            post_play: float | None = None) -> list[Path]:
    pre_play = _padding(cfg.cut_pre_play if pre_play is None else pre_play, "pre-play")
    post_play = _padding(cfg.cut_post_play if post_play is None else post_play, "post-play")
    for play in plays:
        export_bounds(play.start, play.end, clips[-1].end if clips else 0., pre_play, post_play)
    names = (pattern_names(len(plays), name_pattern, start_number, number_width)
             if name_pattern is not None else [name_for(p) for p in plays])
    out_dir = out_dir or cfg.path(cfg.out_dir) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "exports.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    if not isinstance(previous, list) or not all(isinstance(n, str) for n in previous):
        raise ValueError(f"invalid export manifest: {manifest}")

    def check_destinations() -> None:
        if name_pattern is None:
            return
        for name in names:
            target = out_dir / name
            if target.is_symlink() or (target.exists() and
                                      (name not in previous or not target.is_file())):
                raise ValueError(f"destination already exists: {target}")

    check_destinations()
    with tempfile.TemporaryDirectory(prefix=".export-", dir=out_dir) as directory:
        stage = Path(directory)
        total = len(plays)
        failures = []

        def job(p: Play, name: str) -> Path:
            return cut_one(cfg, clips, p, stage / name, pre_play=pre_play, post_play=post_play)

        with cf.ThreadPoolExecutor(max_workers=cfg.clip_workers) as ex:
            futs = {ex.submit(job, p, name): p for p, name in zip(plays, names)}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                p = futs[fut]
                try:
                    fut.result()
                except Exception as exc:
                    failures.append(f"play {p.index}: {exc}")
                if progress:
                    print(f"\r  cutting {i}/{total} clips", end="", flush=True)
        if progress:
            print()
        if failures:
            raise RuntimeError("clip export failed: " + "; ".join(failures))
        check_destinations()
        current = names
        if partial:
            current = sorted(set(previous) | set(current))
        # New custom names are published without replacement, so a file that
        # appears after preflight cannot be overwritten. Claim all new names
        # before replacing any manifest-owned clips or removing stale exports.
        claimed = []
        try:
            if name_pattern is not None:
                for name in names:
                    if name not in previous:
                        target = out_dir / name
                        target.hardlink_to(stage / name)
                        claimed.append(target)
        except OSError:
            for target in claimed:
                target.unlink()
            raise
        for name in names:
            if name_pattern is None or name in previous:
                (stage / name).replace(out_dir / name)
        for name in set(previous) - set(current):
            if Path(name).name == name and name.lower().endswith(".mp4"):
                (out_dir / name).unlink(missing_ok=True)
        temporary = out_dir / ".exports.json.tmp"
        temporary.write_text(json.dumps(current, indent=2))
        temporary.replace(manifest)
    return sorted(out_dir / name for name in current)
