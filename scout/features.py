"""Per-frame signals describing what the players on the field are doing.

Counting people is not enough to recognise a play: during a break an empty
field still shows figures on the touchline, and a post-game handshake puts
twenty people on the pitch.  What separates a play is a formation that holds
still and then breaks together - and, crucially, that a huddle breaking up looks
almost identical unless you measure the *shape* of the motion as well as its
size.

So every frame is described by a vector rather than a handful of numbers.  Some
of it feeds the rule-based segmenter; all of it feeds the learned one.  The
groups are:

  presence    how many players, and how far away they are
  motion      how fast they are going, at several thresholds so nothing hinges
              on one hard cut
  geometry    how they are arranged - including the gap that a line of
              scrimmage puts between two teams
  coherence   whether they move *apart* (a huddle breaking) or *together*
              (a snap).  This is the signal a threshold chain cannot express.
  context     how long it has been quiet

Speeds are in body heights per second, so a player at the far touchline and one
in the foreground are judged the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .detect import Detections
from .track import Tracker, TimeScaledTracker

#: Speeds (body heights / second) at which player counts are taken.  Several
#: thresholds mean the model can find its own cut rather than inheriting mine.
SPEED_LEVELS = (0.4, 0.6, 0.8, 1.2)

FEATURE_NAMES: list[str] = [
    # presence
    "n_field", "n_active", "height_med", "depth_med", "depth_far",
    # motion
    *[f"n_over_{s:g}".replace(".", "") for s in SPEED_LEVELS],
    "n_sprint", "spd_p25", "spd_p50", "spd_p75", "spd_p90", "spd_top5",
    # geometry
    "x_span", "x_std", "y_span", "y_std", "hull", "los_gap", "los_balance",
    # coherence
    "divergence", "coherence", "x_span_rate",
    # context
    "quiet_age",
]


@dataclass
class Features:
    """A (frames x features) matrix plus the clock, addressable by name."""
    t: np.ndarray                 # (F,) global seconds
    X: np.ndarray                 # (F, D)
    names: list[str]

    def __len__(self) -> int:
        return len(self.t)

    def col(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]

    def __getattr__(self, name: str) -> np.ndarray:
        # Lets the rule-based segmenter keep using f.n_sprint, f.x_span etc.
        names = self.__dict__.get("names")
        if names and name in names:
            return self.X[:, names.index(name)]
        raise AttributeError(name)


def _foot_points(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground contact point and apparent height for each box."""
    x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    y = boxes[:, 3]
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    return np.stack([x, y], 1), h


def _on_field(boxes: np.ndarray, mask: np.ndarray, cfg: Config) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, bool)
    H, W = mask.shape
    pts, h = _foot_points(boxes)
    xi = pts[:, 0].astype(int).clip(0, W - 1)
    yi = pts[:, 1].astype(int).clip(0, H - 1)
    return mask[yi, xi] & (h > cfg.min_person_h * H) & (h < cfg.max_person_h * H)


def field_axis(det, mask, cfg: Config, step: int = 20) -> np.ndarray:
    """The direction the field runs, in image coordinates.

    Everything about the arrangement of players - how spread out they are, the
    gap a line of scrimmage leaves - is measured along the field, not along the
    image.  Those happen to coincide when the camera looks straight down the
    touchline, which is true of every venue so far, and would quietly stop being
    true the first time a camera sits at an angle.

    Players move up and down the field far more than across it, so the principal
    axis of where they are found is the field's long axis.
    """
    pts = []
    for i in range(0, len(det), step):
        boxes, _ = det.frame(i)
        b = boxes[_on_field(boxes, mask, cfg)]
        if len(b):
            pts.append(_foot_points(b)[0])
    if not pts:
        return np.array([1.0, 0.0])
    P = np.concatenate(pts)
    if len(P) < 10:
        return np.array([1.0, 0.0])
    P = P - P.mean(0)
    # Perspective squashes depth, so weight both directions equally before
    # asking which way the spread points.
    scale = P.std(0)
    scale[scale < 1e-6] = 1.0
    _, _, vt = np.linalg.svd(P / scale, full_matrices=False)
    axis = vt[0] * scale
    axis = axis / (np.linalg.norm(axis) or 1.0)
    if axis[0] < 0:
        axis = -axis
    return axis


def _project(pos: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Positions as (along the field, across it)."""
    if len(pos) == 0:
        return pos
    perp = np.array([-axis[1], axis[0]])
    return np.stack([pos @ axis, pos @ perp], 1)


def _geometry(pos: np.ndarray, W: int, H: int) -> tuple[float, ...]:
    """Arrangement of the players: extent, and the gap between two groups."""
    if len(pos) < 2:
        return (0.0,) * 7
    xs, ys = pos[:, 0], pos[:, 1]
    x_span = float(xs.max() - xs.min()) / W
    y_span = float(ys.max() - ys.min()) / H
    x_std, y_std = float(xs.std()) / W, float(ys.std()) / H

    hull = 0.0
    if len(pos) >= 3:
        try:
            hull = float(cv2.contourArea(
                cv2.convexHull(pos.astype(np.float32)))) / (W * H)
        except cv2.error:
            hull = 0.0

    # Two teams on a line of scrimmage leave one wide gap across the field; a
    # huddle leaves none.  Take the widest gap and how evenly it splits them.
    order = np.sort(xs)
    gaps = np.diff(order)
    if len(gaps):
        k = int(np.argmax(gaps))
        los_gap = float(gaps[k]) / W
        left, right = k + 1, len(order) - k - 1
        los_balance = min(left, right) / max(left, right)
    else:
        los_gap = los_balance = 0.0
    return x_span, x_std, y_span, y_std, hull, los_gap, los_balance


def _coherence(pos: np.ndarray, vel: np.ndarray) -> tuple[float, float]:
    """Are they moving apart, or moving together?

    ``divergence`` is how much the motion points away from the group's centre:
    near +1 when a huddle breaks up, near 0 for a snap, where players run
    downfield rather than outward.  ``coherence`` is how aligned the directions
    are: near 1 when everyone goes the same way, near 0 when it is a scramble.
    """
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 1e-6
    if moving.sum() < 2:
        return 0.0, 0.0
    v = vel[moving] / speed[moving, None]
    radial = pos[moving] - pos[moving].mean(0)
    rnorm = np.linalg.norm(radial, axis=1)
    ok = rnorm > 1e-6
    divergence = float((v[ok] * (radial[ok] / rnorm[ok, None])).sum(1).mean()) \
        if ok.any() else 0.0
    coherence = float(np.linalg.norm(v.mean(0)))
    return divergence, coherence


def compute(cfg: Config, det: Detections, mask: np.ndarray, *, tracker_factory=None) -> Features:
    """Feature matrix for one camera file; optional factory supports isolated experiments."""
    dt = 1.0 / cfg.sample_fps
    if cfg.feature_pipeline not in ('legacy','dynamic-field-v1'):
        raise ValueError('Unsupported feature pipeline')
    factory=tracker_factory or (TimeScaledTracker if cfg.feature_pipeline=='dynamic-field-v1' else Tracker)
    tracker = factory(dt=dt)
    H, W = mask.shape
    axis = field_axis(det, mask, cfg) if cfg.field_axis else np.array([1.0, 0.0])
    F = len(det)
    X = np.zeros((F, len(FEATURE_NAMES)), np.float32)

    prev_x_span = 0.0
    quiet_age = 0.0

    for i in range(F):
        boxes, _ = det.frame(i)
        b = boxes[_on_field(boxes, mask, cfg)]
        pts, h = _foot_points(b)
        live = tracker.update(pts, h)

        # Only people who have run recently count as players; the rest are
        # standing on the touchline watching.
        act = [t for t in live if t.age >= 2 and t.recent_max > cfg.active_speed]
        pos = np.array([t.pos for t in act]) if act else np.zeros((0, 2))
        vel = np.array([t.vel for t in act]) if act else np.zeros((0, 2))
        spd = np.array([t.speed for t in act]) if act else np.zeros(0)

        counts = [float((spd > s).sum()) for s in SPEED_LEVELS]
        n_sprint = float((spd > cfg.sprint_speed).sum())
        if len(spd):
            p25, p50, p75, p90 = np.percentile(spd, [25, 50, 75, 90])
            top5 = float(np.sort(spd)[-5:].mean())
        else:
            p25 = p50 = p75 = p90 = top5 = 0.0

        geo = _geometry(_project(pos, axis), W, H)
        div, coh = _coherence(pos, vel)   # already rotation-independent

        # A huddle spreads out fast; a formation is already spread.
        x_span_rate = (geo[0] - prev_x_span) / dt
        prev_x_span = geo[0]

        quiet_age = 0.0 if n_sprint >= 2 else min(quiet_age + dt, 60.0)

        X[i] = [
            float(len(live)), float(len(act)),
            float(np.median(h)) / H if len(h) else 0.0,
            float(np.median(pos[:, 1])) / H if len(pos) else 0.0,
            float(np.percentile(pos[:, 1], 10)) / H if len(pos) else 0.0,
            *counts, n_sprint, p25, p50, p75, p90, top5,
            *geo, div, coh, x_span_rate, quiet_age,
        ]

    return Features(det.times.copy(), X, list(FEATURE_NAMES))


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded so the length is preserved."""
    if window <= 1:
        return x.astype(np.float32)
    k = np.ones(window, np.float32) / window
    pad = window // 2
    return np.convolve(np.pad(x.astype(np.float32), pad, mode="edge"),
                       k, "valid")[:len(x)]
