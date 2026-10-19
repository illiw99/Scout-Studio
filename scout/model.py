"""Learned play detection.

The rule-based segmenter asks a fixed question - was there a formation, did it
go still, did it break - with thresholds set by hand.  That got the pipeline
working, but the right thresholds differ from game to game: a file with long
gaps between plays needs strict settings, a busy one needs loose settings, and
no single set of constants can be both.

This asks a different question.  For every sampled frame, how likely is it that
a play is in progress?  That turns a handful of marked plays into tens of
thousands of labelled frames, and lets one model weigh the signals against each
other instead of passing them through a chain of hard cuts.  Plays are then read
off the probability curve, which also gives their start and end without a
separate rule for each.

Each frame is described by its own features plus the same features a few seconds
either side, so the model sees the *shape* of a snap - still, then moving -
rather than only the instant.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field as _field
from pathlib import Path

import numpy as np

from .config import Config
from .features import Features

#: Seconds either side of a frame that the model is allowed to look.  A snap is
#: recognised by what happens around it, not at a single instant.
OFFSETS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


@dataclass
class Dataset:
    X: np.ndarray             # (N, D)
    y: np.ndarray             # (N,) 1 inside a marked play
    groups: np.ndarray        # (N,) index of the file each row came from
    t: np.ndarray             # (N,) seconds within that file
    files: list[str]
    names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        return (f"{len(self):,} frames from {len(self.files)} file(s), "
                f"{self.y.mean():.1%} inside a play, {self.X.shape[1]} features")


def window(X: np.ndarray, sample_fps: float,
           offsets: tuple[float, ...] = OFFSETS) -> np.ndarray:
    """Stack each frame's features with those from nearby frames.

    Edges repeat the first and last frame rather than padding with zeros, so a
    play at the very start of a file is not described by fictitious stillness.
    """
    n = len(X)
    steps = [int(round(o * sample_fps)) for o in offsets]
    idx = np.arange(n)
    cols = [X[np.clip(idx + s, 0, n - 1)] for s in steps]
    return np.concatenate(cols, axis=1)


def window_names(names: list[str],
                 offsets: tuple[float, ...] = OFFSETS) -> list[str]:
    return [f"{nm}@{o:+g}s" for o in offsets for nm in names]


def label(t: np.ndarray, marks) -> np.ndarray:
    """1 for frames inside a marked play."""
    y = np.zeros(len(t), np.int8)
    for m in marks:
        y[(t >= m.start) & (t <= m.end)] = 1
    return y


def build(cfg: Config, per_file: list[tuple[str, Features, list]]) -> Dataset:
    """Assemble the training set from files that have been marked by hand."""
    Xs, ys, gs, ts, names = [], [], [], [], None
    files = []
    drop = set(cfg.model_exclude or ())
    keep = [i for i, nm in enumerate(per_file[0][1].names) if nm not in drop]
    for gi, (stem, f, marks) in enumerate(per_file):
        Xw = window(f.X[:, keep], cfg.sample_fps)
        if names is None:
            names = window_names([f.names[i] for i in keep])
        Xs.append(Xw)
        ys.append(label(f.t, marks))
        gs.append(np.full(len(f.t), gi))
        ts.append(f.t)
        files.append(stem)
    return Dataset(np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
                   np.concatenate(ts), files, names or [])


def make_classifier(cfg: Config):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Deliberately small.  Ten files from two venues is not much variety, and a
    # big model would memorise this tournament rather than learn the game.
    return HistGradientBoostingClassifier(
        max_depth=cfg.model_max_depth,
        max_iter=cfg.model_max_iter,
        learning_rate=cfg.model_learning_rate,
        min_samples_leaf=cfg.model_min_leaf,
        l2_regularization=cfg.model_l2,
        early_stopping=False,
        random_state=0,
    )


def fit(cfg: Config, ds: Dataset, rows: np.ndarray | None = None):
    clf = make_classifier(cfg)
    X = ds.X if rows is None else ds.X[rows]
    y = ds.y if rows is None else ds.y[rows]
    if len(np.unique(y)) != 2:
        raise ValueError("training needs both reviewed plays and non-play footage in each training fold")
    clf.fit(X, y)
    return clf


def predict(clf, f: Features, cfg: Config) -> np.ndarray:
    """P(a play is in progress) for every frame of one file."""
    drop = set(cfg.model_exclude or ())
    keep = [i for i, nm in enumerate(f.names) if nm not in drop]
    return clf.predict_proba(window(f.X[:, keep], cfg.sample_fps))[:, 1]


# --- turning probabilities back into plays --------------------------------

def intervals(cfg: Config, t: np.ndarray, prob: np.ndarray) -> list[tuple[float, float]]:
    """Read plays off the probability curve.

    Hysteresis, so a dip mid-play does not split it in two: a play opens where
    the probability is convincing and stays open while it is merely plausible.
    """
    from .features import smooth
    dt = 1.0 / cfg.sample_fps
    p = smooth(prob, max(1, int(round(cfg.model_smooth / dt))))

    hi, lo = cfg.model_threshold, cfg.model_threshold * cfg.model_low_ratio
    out: list[list[float]] = []
    open_at = None
    for i, v in enumerate(p):
        if open_at is None:
            if v >= hi:
                j = i
                while j > 0 and p[j - 1] >= lo:
                    j -= 1
                open_at = j
        elif v < lo:
            out.append([open_at, i - 1])
            open_at = None
    if open_at is not None:
        out.append([open_at, len(p) - 1])

    # Join plays split by a momentary dip, then drop anything too short to be a
    # play at all.
    merged: list[list[float]] = []
    gap = cfg.merge_gap / dt
    for a, b in out:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    spans = []
    for a, b in merged:
        t0, t1 = float(t[int(a)]), float(t[int(b)])
        if t1 - t0 >= cfg.model_min_duration:
            spans.append((t0, t1))
    return spans


def to_plays(cfg: Config, spans: list[tuple[float, float]], t_end: float | None):
    from .segment import Play
    plays = []
    for i, (a, b) in enumerate(spans, 1):
        end = b + cfg.model_post_roll
        if t_end is not None:
            end = min(end, t_end)
        plays.append(Play(
            index=i, start=max(0.0, a - cfg.model_pre_roll), end=end,
            snap=a, whistle=b, duration=b - a,
            peak_action=0.0, peak_sprinters=0, median_players=0,
        ))
    return plays


# --- persistence ----------------------------------------------------------

def save(path: Path, clf, names: list[str], cfg: Config) -> None:
    """Store the model together with the shape of input it expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fh:
        temporary = Path(fh.name)
        pickle.dump({"clf": clf, "names": names, "sample_fps": cfg.sample_fps,
                     "offsets": OFFSETS, "exclude": tuple(cfg.model_exclude or ()),
                     "n_features": len(names)}, fh)
    temporary.replace(path)


def resolve(path: str | Path) -> Path:
    """Find a model whether or not the command was run from the project root.

    Looked for where it was asked for first, then next to the installed
    package, so an installed copy carries a working model with it.
    """
    p = Path(path)
    if p.exists():
        return p
    here = Path(__file__).parent / p.name
    return here if here.exists() else p


def load(path: Path, cfg: Config | None = None):
    """Load a model, refusing one that was trained on different inputs.

    Changing which features are excluded, or the sample rate, silently changes
    the width of every row handed to the model.  Left unchecked that is either a
    crash or - worse - a confident answer computed from columns that no longer
    mean what the model thinks they mean.
    """
    path = resolve(path)
    with Path(path).open("rb") as fh:
        d = pickle.load(fh)
    if cfg is not None:
        want = tuple(cfg.model_exclude or ())
        got = tuple(d.get("exclude", ()))
        if set(want) != set(got):
            raise SystemExit(
                f"{path} was trained with model_exclude={got or '()'}, but the "
                f"config now says {want or '()'}.\n"
                f"  Re-run 'scout train' so the model matches the settings.")
        if abs(d.get("sample_fps", cfg.sample_fps) - cfg.sample_fps) > 1e-6:
            raise SystemExit(
                f"{path} was trained at {d['sample_fps']} fps, config says "
                f"{cfg.sample_fps}.  Re-run 'scout train'.")
    return d["clf"], d["names"]


def importances(clf, names: list[str], X: np.ndarray, y: np.ndarray,
                top: int = 15, sample: int = 8000,
                repeats: int = 3, seed: int = 0) -> list[tuple[str, float]]:
    """How much the model actually depends on each signal.

    Permuting one column at a time is useless here: every feature appears at
    nine time offsets, so shuffling one leaves eight copies of the same
    information and the score barely moves.  All offsets of a feature are
    therefore shuffled together, which measures what removing that signal
    entirely would cost.

    Worth reading rather than skipping.  If the model leans on something with no
    physical meaning - an absolute position, say - then it has learned these
    particular camera angles and the held-out score is luck.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(sample, len(y)), replace=False)
    Xs, ys = X[idx].copy(), y[idx]
    base = average_precision_score(ys, clf.predict_proba(Xs)[:, 1])

    groups: dict[str, list[int]] = {}
    for i, nm in enumerate(names):
        groups.setdefault(nm.split("@")[0], []).append(i)

    out = []
    for g, cols in groups.items():
        drops = []
        for r in range(repeats):
            Xp = Xs.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]      # keep rows intact within group
            drops.append(base - average_precision_score(
                ys, clf.predict_proba(Xp)[:, 1]))
        out.append((g, float(np.mean(drops))))
    out.sort(key=lambda kv: -kv[1])
    return out[:top]


def grouped_importances(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Already grouped; kept so callers need not care."""
    return pairs
