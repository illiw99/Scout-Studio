"""Working out which part of the frame is the playing surface.

The camera also sees spectators, team benches, neighbouring pitches and a city
skyline, all of which contain people.  The pitch is the one large region of
grass, so its upper boundary is found per column as the height at which green
takes over, and everything below that line is treated as the field.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config

# Artificial turf and grass in HSV.  Wide enough to survive hard sun and the
# deep shade under the trees along the far touchline.
_GREEN_LO = np.array([28, 40, 35], np.uint8)
_GREEN_HI = np.array([95, 255, 255], np.uint8)


def green_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def estimate_mask(frames: list[np.ndarray], cfg: Config) -> np.ndarray:
    """Boolean field mask (H, W) from a handful of sample frames.

    Averaging several frames removes the players themselves, who are green-free
    holes in any single frame.
    """
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w), np.float32)
    for f in frames:
        acc += green_mask(f) > 0
    grass = (acc / len(frames)) > 0.5

    # Per column, the pitch starts at the first row that is grass for a long
    # unbroken run; short runs above are gaps between trees and fences.
    min_run = h // 12
    top = np.full(w, h, np.int32)
    for x in range(w):
        col = grass[:, x]
        run = start = 0
        for y in range(h):
            if col[y]:
                if run == 0:
                    start = y
                run += 1
                if run >= min_run:
                    top[x] = start
                    break
            else:
                run = 0

    # The touchline is straight, so a smooth fit rejects columns confused by a
    # goal net, a tent or a player standing on the line.
    xs = np.arange(w)
    ok = top < h
    if ok.sum() > w // 4:
        med = np.median(top[ok])
        mad = np.median(np.abs(top[ok] - med)) + 1
        good = ok & (np.abs(top - med) < 6 * mad)
        top = np.polyval(np.polyfit(xs[good], top[good], 2), xs)
    else:                                   # no grass found - fall back flat
        top = np.full(w, cfg.horizon_y * h, float)

    # Never trust the fit above the configured horizon: crowds stand there.
    top = np.maximum(top, cfg.horizon_y * h)
    top = top + max(cfg.field_shrink_top, 0.028) * h

    mask = np.arange(h)[:, None] >= top[None, :]
    return mask


def polygon_mask(poly: list[list[float]], shape: tuple[int, int],
                 shrink_top: float = 0.0) -> np.ndarray:
    h, w = shape
    pts = np.array([[p[0] * w, p[1] * h] for p in poly], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    mask = m.astype(bool)
    if shrink_top > 0:
        mask = lower_top_edge(mask, int(round(shrink_top * h)))
    return mask


def lower_top_edge(mask: np.ndarray, px: int) -> np.ndarray:
    """Move only the upper boundary of the mask down by ``px``.

    The far touchline is where substitutes, coaches and spectators stand, so the
    top edge of a hand-drawn field almost always includes people who are not
    playing.  Shrinking from the top alone removes them without trimming the
    ends of the field, where play genuinely reaches the boundary.
    """
    if px <= 0:
        return mask
    out = mask.copy()
    cols = np.where(mask.any(axis=0))[0]
    first = mask.argmax(axis=0)
    for x in cols:
        out[first[x]:first[x] + px, x] = False
    return out


def build(cfg: Config, frames: list[np.ndarray]) -> np.ndarray:
    """Field mask, either from the configured polygon or estimated from grass."""
    if cfg.field_polygon:
        return polygon_mask(cfg.field_polygon, frames[0].shape[:2])
    return estimate_mask(frames, cfg)


def overlay(rgb: np.ndarray, mask: np.ndarray, boxes: np.ndarray | None = None,
            keep: np.ndarray | None = None,
            polygon: list[list[float]] | None = None) -> np.ndarray:
    """Render the mask, and the polygon as drawn, for visual checking.

    These differ whenever ``field_shrink_top`` is non-zero, and showing only the
    effective mask makes it look as though the polygon was drawn somewhere it
    was not.  Cyan is what was drawn; yellow is what is actually used.
    """
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    tint = out.copy()
    tint[mask] = (0, 200, 0)
    out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)
    edge = mask.astype(np.uint8)
    cont, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cont, -1, (0, 255, 255), 2)
    if polygon:
        h, w = mask.shape
        pts = np.array([[p[0] * w, p[1] * h] for p in polygon], np.int32)
        cv2.polylines(out, [pts], True, (255, 220, 0), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts, 1):
            cv2.circle(out, (int(x), int(y)), 5, (255, 220, 0), -1)
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 0), 1)
    if boxes is not None:
        for i, b in enumerate(boxes):
            on = True if keep is None else bool(keep[i])
            col = (0, 0, 255) if on else (140, 140, 140)
            x1, y1, x2, y2 = b[:4].astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if on else 1)
    return out


# --- drift ----------------------------------------------------------------
# The camera creeps a few dozen pixels over an hour, so a polygon picked on one
# frame is nudged to match each file rather than re-picked.

def _small_gray(img: np.ndarray, w: int = 480) -> np.ndarray:
    h = int(img.shape[0] * w / img.shape[1])
    g = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32)


def estimate_shift(ref: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Global (dx, dy) in full-resolution pixels between two framings."""
    a, b = _small_gray(ref), _small_gray(other)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    scale = ref.shape[1] / a.shape[1]
    return dx * scale, dy * scale


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(mask.astype(np.uint8), M,
                         (mask.shape[1], mask.shape[0]),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    return out.astype(bool)


def build_for(cfg: Config, frames: list[np.ndarray],
              ref: np.ndarray | None = None,
              polygon: list[list[float]] | None = None) -> np.ndarray:
    """Field mask for one file.

    With a hand-picked polygon the mask is that polygon, translated by however
    far this file's framing has drifted from the frame it was picked on.  With
    no polygon it falls back to estimating the grass, which finds every pitch in
    view and is only a starting point.
    """
    shape = frames[0].shape[:2]
    poly = polygon if polygon is not None else cfg.field_polygon
    if poly:
        mask = polygon_mask(poly, shape, cfg.field_shrink_top)
        if ref is not None:
            dx, dy = estimate_shift(ref, frames[len(frames) // 2])
            if abs(dx) > 1 or abs(dy) > 1:
                mask = shift_mask(mask, dx, dy)
        return mask
    return estimate_mask(frames, cfg)
