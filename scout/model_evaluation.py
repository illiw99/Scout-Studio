"""Read-only helpers for isolated decision-model experiments."""
from __future__ import annotations

import math
import numpy as np
from .detect import Detections


def subsample(det: Detections, source_fps: float, target_fps: float) -> Detections:
    """Select actual sampled frames; recompute tracking later at target FPS."""
    if not all(math.isfinite(f) and f > 0 for f in (source_fps, target_fps)):
        raise ValueError('Sampling rates must be positive and finite')
    ratio = source_fps / target_fps
    step = round(ratio)
    if step < 1 or not math.isclose(step, ratio, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Target rate must divide the original sampling rate exactly')
    if (len(det.offsets) != len(det.times) + 1 or det.offsets[0] != 0
            or det.offsets[-1] != len(det.boxes) or len(det.conf) != len(det.boxes)
            or np.any(np.diff(det.offsets) < 0)):
        raise ValueError('Invalid detection frame offsets')
    frames = np.arange(0, len(det), step)
    counts = np.diff(det.offsets)[frames]
    indices = np.concatenate([np.arange(det.offsets[i], det.offsets[i + 1]) for i in frames]) if len(frames) else np.empty(0, dtype=int)
    return Detections(det.times[frames].copy(), np.r_[0, np.cumsum(counts)].astype(np.int64),
                      det.boxes[indices].copy(), det.conf[indices].copy(), det.mask_shape)


def interval_metrics(truth, predictions, minimum_iou=.5):
    """Match maximum event count first, then IoU; report boundary errors too."""
    from scipy.optimize import linear_sum_assignment
    if not 0 <= minimum_iou <= 1:
        raise ValueError('IoU threshold must be in [0,1]; zero means positive overlap')
    truth, predictions = np.asarray(truth, dtype=float).reshape(-1, 2), np.asarray(predictions, dtype=float).reshape(-1, 2)
    for intervals in (truth, predictions):
        if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
            raise ValueError('Intervals must have positive finite duration')
    hits, start, end, overlaps = 0, [], [], []
    if len(truth) and len(predictions):
        intersection = np.maximum(0, np.minimum(truth[:, None, 1], predictions[None, :, 1])
                                  - np.maximum(truth[:, None, 0], predictions[None, :, 0]))
        union = np.maximum(truth[:, None, 1], predictions[None, :, 1]) - np.minimum(truth[:, None, 0], predictions[None, :, 0])
        iou = intersection / union
        valid = (intersection > 0) & (iou >= minimum_iou)
        rows, columns = linear_sum_assignment(-(valid * (min(len(truth), len(predictions)) + 1) + np.where(valid, iou, 0)))
        for row, column in zip(rows, columns):
            if valid[row, column]:
                hits += 1
                start.append(float(predictions[column, 0] - truth[row, 0]))
                end.append(float(predictions[column, 1] - truth[row, 1]))
                overlaps.append(float(iou[row, column]))
    return dict(hits=hits, missed=len(truth)-hits, false_positives=len(predictions)-hits,
                start_errors=start, end_errors=end, ious=overlaps)


def summarize(metrics):
    hits = sum(m['hits'] for m in metrics)
    missed = sum(m['missed'] for m in metrics)
    false = sum(m['false_positives'] for m in metrics)
    start = [v for m in metrics for v in m['start_errors']]
    end = [v for m in metrics for v in m['end_errors']]
    ious = [v for m in metrics for v in m['ious']]
    tolerance = {str(t):float(np.mean((np.abs(start)<=t) & (np.abs(end)<=t))) if start else None
                 for t in (.5,1.,2.)}
    extra = [max(0,-a)+max(0,b) for a,b in zip(start,end)]
    truncated = [max(0,a)+max(0,-b) for a,b in zip(start,end)]
    return dict(hits=hits, missed=missed, false_positives=false,
                recall=hits/(hits+missed) if hits+missed else 0,
                precision=hits/(hits+false) if hits+false else 0,
                f1=2*hits/(2*hits+missed+false) if 2*hits+missed+false else 0,
                start_mae=float(np.mean(np.abs(start))) if start else None,
                end_mae=float(np.mean(np.abs(end))) if end else None,
                boundary_p90=float(np.percentile(np.abs(start+end),90)) if start else None,
                both_boundaries_within_seconds=tolerance,
                mean_iou=float(np.mean(ious)) if ious else None,
                extra_seconds_per_detected_play=float(np.mean(extra)) if extra else None,
                truncated_seconds_per_detected_play=float(np.mean(truncated)) if truncated else None)
