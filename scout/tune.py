"""Searching for detector settings that match hand-marked footage.

Thresholds tuned by eye on one file look better than they are.  This scores
every candidate against each marked file separately and reports them side by
side, so a setting that only works on the file it was fitted to is obvious
rather than hidden inside an average.

The search is split in two because the two halves cost very different amounts:
anything that changes which people count as players forces the tracker to run
again, while the thresholds that shape a snap are almost free to vary.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import Config
from . import timeline, detect, features, segment, groundtruth, evaluate

#: Settings that change the per-frame signals, so the tracker must re-run.
FEATURE_GRID = {
    "field_shrink_top": (0.0,),
    "sprint_speed": (0.8, 0.9, 1.0),
    "active_speed": (0.35, 0.45),
    "min_person_h": (0.020, 0.030),
}

#: Settings that only reshape the snap test; cheap to vary.
SEGMENT_GRID = {
    "burst_min": (3, 4, 5),
    "formation_span": (0.10, 0.15, 0.22),
    "still_max": (2, 3, 4),
    "still_hold": (0.6, 0.8, 1.2),
    "min_players": (4, 5, 6),
    "refractory": (8.0, 10.0, 12.0),
}

QUICK = {
    "sprint_speed": (0.9,), "active_speed": (0.45,), "min_person_h": (0.030,),
    "burst_min": (4,), "formation_span": (0.15,), "still_max": (3,),
    "still_hold": (0.8,), "min_players": (6,), "refractory": (10.0,),
}


@dataclass
class Case:
    """One marked file and everything needed to score against it."""
    name: str
    clips: list
    det: object
    gt: list

    @classmethod
    def build(cls, cfg: Config, mlt: str) -> "Case":
        clips = groundtruth.project_clips(mlt)
        if len(clips) != 1:
            raise ValueError("tune supports single-video reviews only; use train for multi-video matches")
        name = clips[0].path.name
        if not clips:
            raise SystemExit(f"{mlt}: {name} not found in {cfg.media_dir}")
        d = detect.load_cached(cfg, clips[0])
        if d is None:
            raise SystemExit(f"{name}: not detected yet - run 'scout detect'")
        return cls(name, clips, d, groundtruth.load(mlt))


def _combos(grid: dict, quick: bool) -> list[dict]:
    g = {k: (QUICK.get(k, v) if quick else v) for k, v in grid.items()}
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*g.values())]


def search(cfg: Config, mlts: list[str], quick: bool = False,
           progress: bool = True) -> list[dict]:
    cases = [Case.build(cfg, m) for m in mlts]
    masks = {}
    regions = {}
    for c in cases:
        from .cli import _region_for
        r = _region_for(cfg, c.clips, c.clips[0])
        regions[c.name] = r

    feat_combos = _combos(FEATURE_GRID, quick)
    seg_combos = _combos(SEGMENT_GRID, quick)
    total = len(feat_combos) * len(seg_combos)
    if progress:
        print(f"  {len(cases)} marked file(s), {len(feat_combos)} feature x "
              f"{len(seg_combos)} segment = {total} combinations")

    results: list[dict] = []
    t0 = time.time()
    for fi, fp in enumerate(feat_combos, 1):
        base = replace(cfg, **fp)
        feats = {}
        for c in cases:
            r = regions[c.name]
            key = (c.name, fp["field_shrink_top"])
            if key not in masks:
                ref = None
                poly = None
                if r:
                    poly = r["polygon"]
                    from .media import grab_frame
                    try:
                        ref = grab_frame(c.clips[0].path, r.get("ref_time", 60.0))
                    except Exception:
                        ref = None
                masks[key] = detect.field_mask(base, c.clips[0], ref, poly)
            feats[c.name] = features.compute(base, c.det, masks[key])

        for sp in seg_combos:
            trial = replace(base, **sp)
            per = {}
            hits = miss = fp_n = 0
            errs = []
            for c in cases:
                plays = segment.find(trial, feats[c.name], t_end=c.clips[-1].end)
                sc = evaluate.score(c.gt, plays)
                per[c.name] = sc
                hits += len(sc.hits); miss += len(sc.missed); fp_n += len(sc.false_pos)
                errs += [abs(m.det.start - m.gt.start) + abs(m.det.end - m.gt.end)
                         for m in sc.hits]
            rec = hits / (hits + miss) if hits + miss else 0.0
            pre = hits / (hits + fp_n) if hits + fp_n else 0.0
            f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0
            results.append({
                "f1": f1, "recall": rec, "precision": pre,
                "boundary": float(np.mean(errs)) if errs else float("inf"),
                "params": {**fp, **sp},
                "per_file": {k: (v.recall, v.precision, v.f1) for k, v in per.items()},
                "worst_f1": min(v.f1 for v in per.values()),
            })
        if progress:
            done = fi * len(seg_combos)
            el = time.time() - t0
            print(f"\r  {done}/{total}  ({el:.0f}s elapsed)", end="", flush=True)
    if progress:
        print()
    return results


def report(results: list[dict], cases: list[str], top: int = 10,
           key: str = "f1") -> dict:
    """Print the best settings and return the winner."""
    ranked = sorted(results, key=lambda r: (-r[key], r["boundary"]))
    print(f"\n  best by {key} (worst_f1 = the weakest single file, "
          f"which is what generalisation looks like)\n")
    head = f"  {'F1':>6} {'recall':>7} {'prec':>6} {'worst':>6} {'bound':>6}  "
    print(head + "  ".join(f"{n[:8]:>8}" for n in cases) + "   params")
    for r in ranked[:top]:
        pf = "  ".join(f"{r['per_file'].get(n, (0,0,0))[2]:8.1%}" for n in cases)
        ps = " ".join(f"{k}={v}" for k, v in r["params"].items()
                      if k not in ("field_shrink_top",))
        print(f"  {r['f1']:6.1%} {r['recall']:7.1%} {r['precision']:6.1%} "
              f"{r['worst_f1']:6.1%} {r['boundary']:6.2f}  {pf}   {ps}")
    return ranked[0]
