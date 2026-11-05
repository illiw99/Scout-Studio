"""Optional diagnostic plots, frames and play tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .features import Features
from .media import grab_frame
from .segment import Play
from .timeline import Clip, locate


def write_tables(plays: list[Play], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plays.json").write_text(
        json.dumps([p.as_dict() for p in plays], indent=2))
    with (out_dir / "plays.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "clip_start_s", "clip_end_s", "snap_s", "whistle_s",
                    "duration_s", "peak_sprinters", "median_players", "truncated"])
        for p in plays:
            w.writerow([p.index, f"{p.start:.2f}", f"{p.end:.2f}", f"{p.snap:.2f}",
                        f"{p.whistle:.2f}", f"{p.duration:.2f}", p.peak_sprinters,
                        p.median_players, int(p.truncated)])


def timeline_plot(cfg: Config, f: Features, plays: list[Play], dst: Path,
                  chunk_minutes: float = 8.0) -> list[Path]:
    """Action signal over time with detected plays shaded.

    Split into rows of a few minutes each, otherwise an hour of footage squeezes
    every play into a couple of pixels.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name == 'matplotlib':
            return []  # Diagnostic plots are optional in the review/export runtime.
        raise
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.clip(f.n_sprint / 8.0, 0, 1)
    t0, t1 = float(f.t[0]), float(f.t[-1])
    span = chunk_minutes * 60
    rows = int(np.ceil((t1 - t0) / span))
    fig, axes = plt.subplots(rows, 1, figsize=(19, 2.1 * rows), squeeze=False)
    for r in range(rows):
        ax = axes[r][0]
        lo, hi = t0 + r * span, min(t1, t0 + (r + 1) * span)
        sel = (f.t >= lo) & (f.t <= hi)
        ax.plot(f.t[sel], a[sel], lw=1.0, color="#1f77b4")
        ax.plot(f.t[sel], f.x_span[sel], lw=0.8, color="#bbbbbb", zorder=0)
        ax.axhline(cfg.formation_span, color="#999", lw=0.7, ls=":")
        ax.axhline(cfg.burst_min / 8.0, color="#d62728", lw=0.7, ls="--")
        for p in plays:
            if p.whistle < lo or p.snap > hi:
                continue
            ax.axvspan(p.snap, p.whistle, color="#2ca02c", alpha=0.25)
            ax.annotate(str(p.index), (p.snap, 1.04), fontsize=7,
                        color="#2ca02c", ha="center")
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1.15)
        ax.set_ylabel("action"); ax.grid(alpha=0.25)
    axes[-1][0].set_xlabel("time in recording (s)")
    fig.suptitle(f"{len(plays)} plays detected  "
                 f"(blue = players running / 8, grey = formation width, "
                 f"green = play)", y=0.999)
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=80)
    plt.close(fig)
    return [dst]


def filmstrip(clips: list[Clip], play: Play, dst: Path, n: int = 6,
              width: int = 420) -> Path | None:
    """A row of frames spanning the play, for scanning without playback."""
    ts = np.linspace(play.snap - 1.0, play.whistle + 0.5, n)
    tiles = []
    for t in ts:
        c, local = locate(clips, float(max(0.0, t)))
        try:
            fr = grab_frame(c.path, local)
        except Exception:
            continue
        h, w = fr.shape[:2]
        crop = fr[int(0.26 * h):int(0.60 * h)]
        scale = width / crop.shape[1]
        tiles.append(cv2.resize(crop, (width, int(crop.shape[0] * scale))))
    if not tiles:
        return None
    strip = np.hstack(tiles)
    strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    cv2.rectangle(strip, (0, 0), (250, 26), (0, 0, 0), -1)
    cv2.putText(strip, f"#{play.index}  {play.duration:.1f}s", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), strip, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return dst


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def annotate(cfg: Config, clips: list[Clip], t: float, masks: dict,
             dets: dict, warmup: float = 4.0, polygons: dict | None = None
             ) -> np.ndarray:
    """One frame with the field outline, tracked players and their speeds.

    The tracker is run from a few seconds earlier so velocities have settled by
    the time the requested frame is reached.
    """
    from .track import Tracker
    from .features import _foot_points, _on_field
    from . import field as _fieldmod

    clip, local = locate(clips, t)
    det = dets[clip.path.name]
    mask = masks[clip.path.name]

    i_end = int(np.clip(np.searchsorted(det.times, t), 0, len(det) - 1))
    i_start = max(0, i_end - int(warmup * cfg.sample_fps))

    tracker = Tracker(dt=1.0 / cfg.sample_fps)
    live = []
    for i in range(i_start, i_end + 1):
        boxes, _ = det.frame(i)
        b = boxes[_on_field(boxes, mask, cfg)]
        pts, h = _foot_points(b)
        live = tracker.update(pts, h)

    frame = grab_frame(clip.path, local)
    boxes, _ = det.frame(i_end)
    keep = _on_field(boxes, mask, cfg)
    poly = (polygons or {}).get(clip.path.name)
    out = _fieldmod.overlay(frame, mask, boxes, keep, poly)

    fast = 0
    for tr in live:
        if tr.age < 2:
            continue
        sp = tr.speed
        if sp > cfg.sprint_speed:
            fast += 1
        x, y = int(tr.pos[0]), int(tr.pos[1])
        col = (0, 0, 255) if sp > cfg.sprint_speed else (0, 200, 255)
        cv2.putText(out, f"{sp:.1f}", (x - 16, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(out, f"{sp:.1f}", (x - 16, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    h = int(frame.shape[0] * 0.76)
    out = out[:h]
    cv2.rectangle(out, (0, 0), (900, 40), (0, 0, 0), -1)
    cv2.putText(out, f"t={t:.1f}s  on-field={int(keep.sum())}  running={fast}"
                     f"  (labels = body-heights/s, red > {cfg.sprint_speed})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if poly:
        cv2.rectangle(out, (0, 40), (560, 74), (0, 0, 0), -1)
        cv2.putText(out, "cyan = polygon you drew", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(out, "| yellow = used", (300, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def snap_sheet(cfg: Config, clips: list[Clip], plays: list[Play], dst: Path,
               cols: int = 4, tile_w: int = 620, offset: float = 0.8) -> Path | None:
    """One tile per play, taken just after the snap.

    The quickest way to judge a whole run: a real snap shows two teams spread
    across the field breaking together, while a false positive shows an empty
    field, a huddle or a group milling about.
    """
    tiles = []
    for p in plays:
        c, local = locate(clips, p.snap + offset)
        try:
            fr = grab_frame(c.path, local)
        except Exception:
            continue
        h, w = fr.shape[:2]
        crop = fr[int(0.27 * h):int(0.58 * h)]
        t = cv2.resize(crop, (tile_w, int(crop.shape[0] * tile_w / crop.shape[1])))
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.rectangle(t, (0, 0), (196, 22), (0, 0, 0), -1)
        cv2.putText(t, f"#{p.index} {int(p.snap)}s {p.duration:.0f}s n={p.median_players}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (60, 60, 60), 1)
        tiles.append(t)
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return dst
