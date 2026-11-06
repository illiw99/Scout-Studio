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


def project_field_region(reference, current, polygon, width=640):
    """Experimental camera projection; return None instead of guessing on failure.

    Does not edit a polygon or any file. Matching is independent of play labels.
    This is an evaluation helper, not a production preprocessing default.
    """
    import cv2
    points=np.asarray(polygon,dtype=np.float32)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all()
            or reference.ndim!=3 or current.ndim!=3 or width<64):
        raise ValueError('Invalid field projection inputs')
    def small(image):
        h,w=image.shape[:2];sw=min(width,w);sh=max(1,round(h*sw/w))
        gray=cv2.cvtColor(cv2.resize(image,(sw,sh)),cv2.COLOR_RGB2GRAY)
        return gray,np.diag([sw/w,sh/h,1.])
    first,scale_ref=small(reference);second,scale_current=small(current)
    detector=cv2.ORB_create(nfeatures=2500)
    k1,d1=detector.detectAndCompute(first,None);k2,d2=detector.detectAndCompute(second,None)
    quality=dict(matches=0,inliers=0,inlier_fraction=0.,accepted=False)
    if d1 is None or d2 is None:return None,quality
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
    quality['matches']=len(good)
    if len(good)<20:return None,quality
    source=np.float32([k1[m.queryIdx].pt for m in good]);target=np.float32([k2[m.trainIdx].pt for m in good])
    matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,3.)
    if matrix is None or inliers is None:return None,quality
    accepted=inliers[:,0].astype(bool);count=int(accepted.sum());fraction=count/len(good)
    quality.update(inliers=count,inlier_fraction=fraction)
    if count<20 or fraction<.5:return None,quality
    spread=np.ptp(source[accepted],axis=0)
    if spread[0]<.3*first.shape[1] or spread[1]<.15*first.shape[0]:return None,quality
    matrix=np.linalg.inv(scale_current)@matrix@scale_ref
    source_pixels=points*[reference.shape[1],reference.shape[0]]
    projected=cv2.perspectiveTransform(source_pixels.astype(np.float32)[None],matrix)[0]
    projected=projected/[current.shape[1],current.shape[0]]
    if not np.isfinite(projected).all() or np.abs(projected).max()>4:return None,quality
    original_area=cv2.contourArea(points,oriented=True)
    projected_area=cv2.contourArea(projected.astype(np.float32),oriented=True)
    if original_area==0 or not .25<=projected_area/original_area<=4:return None,quality
    quality['accepted']=True
    return projected.tolist(),quality
