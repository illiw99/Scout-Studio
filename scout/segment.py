"""Turning the per-frame action signal into a list of plays.

A play is found by its shape rather than by how much movement there is, because
movement alone does not separate a play from a warm-up: during a break players
jog about faster, and in greater numbers, than during a real snap.

What is unique to a snap is the sequence.  Both teams are on the field, spread
across it in a formation; everything goes still for about a second; then several
players break at the same instant.  Nothing else in the footage does all three in
that order.  The play then runs until the movement decays back to a walk.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import Config
from .features import Features


@dataclass
class Play:
    index: int
    start: float          # padded clip start, global seconds
    end: float            # padded clip end, global seconds
    snap: float           # where the action burst began
    whistle: float        # where it decayed
    duration: float       # snap -> whistle
    peak_action: float
    peak_sprinters: int
    median_players: int
    truncated: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _rolling(x: np.ndarray, n: int, how: str) -> np.ndarray:
    """Max or min over the n samples ending at each index."""
    if n <= 1:
        return x.astype(np.float32)
    pad = np.concatenate([np.full(n - 1, x[0], x.dtype), x])
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return win.max(axis=1) if how == "max" else win.min(axis=1)


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling(x, n, "max")


def find_snaps(cfg: Config, f: Features) -> list[int]:
    """Indices where a play starts, by matching the formation-still-break shape."""
    dt = 1.0 / cfg.sample_fps
    n_still = max(1, int(round(cfg.still_hold / dt)))
    n_burst = max(1, int(round(cfg.burst_window / dt)))

    moving = f.n_sprint.astype(np.float32)
    formed = ((f.n_active >= cfg.min_players) &
              (f.x_span >= cfg.formation_span)).astype(np.float32)

    # Busiest moment during the set that precedes each index...
    set_busy = _rolling_max(moving, n_still)
    # ...and whether the teams were spread out across that same window.  This
    # is the maximum, not the minimum: detection drops a player for a frame or
    # two often enough that demanding it hold every frame loses real snaps.
    set_formed = _rolling_max(formed, n_still)

    # Strongest break in the window starting at each index.
    fwd = moving[::-1]
    burst = _rolling_max(fwd, n_burst)[::-1]

    ok = (np.roll(set_busy, 1) <= cfg.still_max) & \
         (np.roll(set_formed, 1) > 0) & \
         (burst >= cfg.burst_min)
    ok[:n_still + 1] = False

    # One snap per play: take the strongest candidate, drop everything within a
    # play's length of it, repeat.  Grouping by proximity instead would let a
    # candidate just outside the group re-fire a few seconds after the snap.
    cand = list(np.flatnonzero(ok))
    gap = int(round(cfg.refractory / dt))
    snaps: list[int] = []
    while cand:
        best = max(cand, key=lambda i: (burst[i], -i))
        snaps.append(int(best))
        cand = [i for i in cand if abs(i - best) >= gap]
    return sorted(snaps)


def _onset(cfg: Config, f: Features, burst: int) -> int:
    """Walk back from the detected burst to where the play actually began.

    A burst is recognised once several players are already past the sprint
    threshold, which is a moment or two after the ball is snapped - measured
    against hand-marked footage, about 2.2 s after.  Walking back to the last
    frame that still looked like the set recovers the real start.
    """
    dt = 1.0 / cfg.sample_fps
    limit = max(0, burst - int(round(cfg.max_backoff / dt)))
    i = burst
    while i > limit and f.n_sprint[i - 1] > cfg.still_max:
        i -= 1
    # Step back over the wind-up too: the frames where movement is building but
    # has not yet reached the threshold.
    while i > limit and f.n_sprint[i - 1] >= 1:
        i -= 1
    return i


def _play_end(cfg: Config, f: Features, start: int) -> int:
    """Where the action after a snap has decayed back to walking."""
    dt = 1.0 / cfg.sample_fps
    hold = max(1, int(round(cfg.end_hold / dt)))
    limit = min(len(f) - 1, start + int(cfg.max_duration / dt))
    quiet = 0
    for i in range(start, limit + 1):
        if f.n_sprint[i] <= 1:
            quiet += 1
            if quiet >= hold:
                return i - hold + 1
        else:
            quiet = 0
    return limit


def find(cfg: Config, f: Features, t_end: float | None = None) -> list[Play]:
    if len(f) == 0:
        return []
    plays: list[Play] = []
    for burst in find_snaps(cfg, f):
        s = _onset(cfg, f, burst)
        e = _play_end(cfg, f, burst)
        snap, whistle = float(f.t[s]), float(f.t[e])
        dur = whistle - snap
        truncated = dur > cfg.max_duration
        if truncated:
            whistle, dur = snap + cfg.max_duration, cfg.max_duration
        if dur < cfg.min_duration:
            whistle, dur = snap + cfg.min_duration, cfg.min_duration
        seg = slice(s, max(s + 1, e + 1))
        start = max(0.0, snap - cfg.pre_roll)
        end = whistle + cfg.post_roll
        if t_end is not None:
            end = min(end, t_end)
        plays.append(Play(
            index=0, start=start, end=end, snap=snap, whistle=whistle,
            duration=dur, peak_action=float(f.x_span[seg].max()),
            peak_sprinters=int(f.n_sprint[seg].max()),
            median_players=int(np.median(f.n_active[seg])), truncated=truncated,
        ))
    plays.sort(key=lambda p: p.start)
    for i, p in enumerate(plays, 1):
        p.index = i
    return plays
