"""Visual-only, numeric boundary-window scoring; no training dependencies."""
from __future__ import annotations

import numpy as np

FORMAT = 'scout-boundary-window-v1'
OFFSETS = np.arange(-6., 6.001, .4)
PROBABILITY_OFFSETS = np.asarray([-3., -2., -1., -.4, 0., .4, 1., 2., 3.])


def feature_names(names):
    return ([prefix + name for prefix in ('center:', 'change:', 'curvature:') for name in names]
            + [f'probability:{offset:g}' for offset in PROBABILITY_OFFSETS] + ['duration', 'shift'])


def nearest(t, positions):
    """Nearest sample, preferring the earlier sample on exact ties."""
    right = np.searchsorted(t, positions).clip(0, len(t)-1)
    left = np.maximum(0, right-1)
    return np.where(np.abs(t[left]-positions) <= np.abs(t[right]-positions), left, right)


class BoundaryRefiner:
    def __init__(self, document, names):
        from .classifier_runtime import Classifier, FORMAT as TREE_FORMAT
        if (not isinstance(document, dict) or set(document) != {'format', 'names', 'heads'}
                or document.get('format') != FORMAT or document.get('names') != list(names)
                or not isinstance(document.get('heads'), list) or len(document['heads']) != 2):
            raise ValueError('Invalid boundary refiner contract')
        expected = feature_names(names)
        self.width = len(names)
        self.heads = []
        for head in document['heads']:
            if (not isinstance(head, dict) or set(head) != {'format','names','baseline','trees'}
                    or head.get('format') != TREE_FORMAT or head.get('names') != expected):
                raise ValueError('Boundary classifier feature schema differs')
            self.heads.append(Classifier(head))

    def refine(self, t, X, probability, proposals, duration):
        t, X, probability = np.asarray(t), np.asarray(X), np.asarray(probability)
        original = np.asarray(proposals, dtype=float).reshape(-1, 2)
        if (t.ndim != 1 or probability.shape != t.shape or X.shape != (len(t), self.width)
                or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.isinf(X).any()
                or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
                or not np.isfinite(duration) or duration <= 0
                or not np.isfinite(original).all() or np.any(original[:,0] < 0)
                or np.any(original[:,1] > duration) or np.any(original[:,1] <= original[:,0])
                or np.any(np.diff(original[:,0]) < 0)):
            raise ValueError('Invalid boundary observations or timeline')
        if not len(original):
            return []
        if not len(t):
            raise ValueError('Boundary proposals require observations')
        shifts = []
        for edge, head in enumerate(self.heads):
            positions = original[:,edge,None] + OFFSETS
            starts = positions if edge == 0 else original[:,0,None]
            ends = original[:,1,None] if edge == 0 else positions
            valid = (starts >= 0) & (ends <= duration) & (ends-starts >= 1.)
            rows, columns = np.nonzero(valid)
            empty = ~valid.any(axis=1)
            if not len(rows):
                shifts.append(np.zeros(len(original)))
                continue
            where = positions[rows,columns]
            center = X[nearest(t,where)]
            before, after = X[nearest(t,where-1)], X[nearest(t,where+1)]
            probabilities = np.column_stack([np.interp(where+offset,t,probability) for offset in PROBABILITY_OFFSETS])
            lengths = ends-starts
            edge_values = np.column_stack([center,after-before,(after+before)/2-center,
                probabilities,lengths[rows,columns]]).astype(np.float32)
            values = np.column_stack([edge_values,OFFSETS[columns]])
            scores = np.full(valid.shape,-np.inf)
            scores[rows,columns] = head.predict_proba(values)[:,1] - 1e-6*np.abs(OFFSETS[columns])
            scores[empty,np.argmin(np.abs(OFFSETS))] = 0
            shifts.append(OFFSETS[np.argmax(scores,axis=1)])
        result = original + np.column_stack(shifts)
        result[:,0] = np.maximum(0,result[:,0]); result[:,1] = np.minimum(duration,result[:,1])
        bad = result[:,1]-result[:,0] < 1.
        result[bad] = original[bad]
        # Independently scored edges must not reverse plays or introduce a new
        # overlap. Keep explicit overlaps already present in the base recipe.
        reverted = bad.copy()
        for _ in range(len(original)):
            conflicts = ((result[1:,0] < result[:-1,0])
                         | ((original[1:,0] >= original[:-1,1])
                            & (result[1:,0] < result[:-1,1]-1e-9)))
            undo = np.r_[conflicts,False] | np.r_[False,conflicts]
            if not np.any(undo & ~reverted):break
            reverted |= undo
            result[undo] = original[undo]
        return [tuple(row) for row in result]
