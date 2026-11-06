"""Cheap, versioned play features derived from the existing player observations."""
from __future__ import annotations

import math
import numpy as np

DEFAULT_OFFSETS = (-3., -2., -1., -.5, 0., .5, 1., 2., 3.)
BASE = 'base-v1'
CONTEXT = 'play-context-v1'
FEATURE_SETS = (BASE, CONTEXT)
EXTRA_NAMES = ('active_fraction', 'sprint_fraction', 'fast_fraction',
               'formation_density', 'activity_rise', 'activity_drop',
               'spread_rise', 'motion_disorder', 'relative_span_change')


def validate_offsets(values):
    if (not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 25
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 30 for v in values)
            or 0 not in values or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError('Invalid classifier temporal offsets')
    return tuple(float(v) for v in values)


def names(raw_names, feature_set=BASE):
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unsupported classifier feature set')
    result = list(raw_names) + (list(EXTRA_NAMES) if feature_set == CONTEXT else [])
    if len(result) != len(set(result)):
        raise ValueError('Duplicate classifier feature names')
    return result


def _neighbors(values, steps):
    """Past/future means excluding this sample, repeating observed edge values."""
    padded = np.pad(values, (steps, steps), mode='edge')
    valid = np.isfinite(padded)
    sums = np.r_[0., np.cumsum(np.where(valid, padded, 0.), dtype=np.float64)]
    counts = np.r_[0, np.cumsum(valid)]
    index = np.arange(len(values))
    def average(start):
        count = counts[start + steps] - counts[start]
        return np.divide(sums[start + steps] - sums[start], count,
                         out=np.full(len(values), np.nan), where=count > 0)
    return average(index), average(index + steps + 1)


def expand(X, raw_names, sample_fps, feature_set=BASE):
    expanded_names = names(raw_names, feature_set)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(raw_names):
        raise ValueError('Player feature shape differs from its schema')
    if feature_set == BASE:
        return X, expanded_names
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError('Invalid classifier sampling rate')
    if not len(X):
        return np.empty((0, len(expanded_names)), dtype=X.dtype), expanded_names
    lookup = {name: X[:, i] for i, name in enumerate(raw_names)}
    try:
        count = np.maximum(lookup['n_field'], 1.)
        def fraction(name):
            return np.where(lookup['n_field'] > 0, np.clip(lookup[name] / count, 0., 1.), 0.)
        active, sprint, fast = (fraction(name) for name in ('n_active', 'n_sprint', 'n_over_08'))
        hull = lookup['hull']
        density = np.log1p(np.maximum(lookup['n_field'], 0.) / np.maximum(hull, .001))
        past, future = _neighbors(active, max(1, round(4 * sample_fps)))
        past_hull, _ = _neighbors(hull, max(1, round(4 * sample_fps)))
        disorder = active * (1. - np.clip(lookup['coherence'], 0., 1.))
        relative_span = lookup['x_span_rate'] / np.maximum(lookup['x_span'], .01)
    except KeyError as exc:
        raise ValueError('Missing player observation for play-context features') from exc
    values = np.column_stack((X, active, sprint, fast, density, active - past,
                              active - future, hull - past_hull, disorder, relative_span))
    return values.astype(np.result_type(X.dtype, np.float32), copy=False), expanded_names
