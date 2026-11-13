"""Validated numeric inference for exported binary histogram-boosted trees.

This module imports NumPy only. It never unpickles models or imports training
frameworks. Export is deliberately limited to numeric binary HGB classifiers.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from .model_features import BASE, DEFAULT_OFFSETS, FEATURE_SETS, validate_offsets

FORMAT = 'scout-hgb-v1'
DYNAMIC_FORMAT = 'scout-hgb-v2'
CONTEXT_FORMAT = 'scout-hgb-v3'
BOUNDARY_FORMAT = 'scout-hgb-v4'
FEATURE_SETTINGS=('min_person_h','max_person_h','active_speed','sprint_speed','field_axis','field_shrink_top','horizon_y')
TREE_FIELDS = {'feature', 'threshold', 'left', 'right', 'value', 'missing_left', 'leaf'}


def _threshold(value):
    if value in ('+inf', '-inf'):
        return float('inf') if value == '+inf' else float('-inf')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid numeric classifier threshold')
    return float(value)


class Classifier:
    def __init__(self, document):
        if not isinstance(document, dict) or document.get('format') not in (FORMAT,DYNAMIC_FORMAT,CONTEXT_FORMAT,BOUNDARY_FORMAT):
            raise ValueError('Unsupported exported classifier format')
        pipeline=document.get('feature_pipeline','legacy')
        if ((document['format']==FORMAT and pipeline!='legacy')
                or (document['format'] in (DYNAMIC_FORMAT,CONTEXT_FORMAT,BOUNDARY_FORMAT) and pipeline!='dynamic-field-v1')):
            raise ValueError('Unsupported classifier feature pipeline')
        if document['format'] in (CONTEXT_FORMAT,BOUNDARY_FORMAT):
            if document.get('feature_set') not in FEATURE_SETS:
                raise ValueError('Unsupported classifier feature set')
            validate_offsets(document.get('offsets'))
        elif document.get('feature_set', BASE) != BASE:
            raise ValueError('Custom classifier features require a version-three classifier')
        names = document.get('names')
        if (not isinstance(names, list) or not names or len(names) > 10000
                or any(not isinstance(name, str) or not name or len(name) > 1024 for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Invalid classifier feature schema')
        baseline = document.get('baseline')
        if type(baseline) not in (int, float) or not math.isfinite(baseline):
            raise ValueError('Invalid classifier baseline')
        trees = document.get('trees')
        if not isinstance(trees, list) or not trees or len(trees) > 2000:
            raise ValueError('Invalid classifier tree count')
        self.names, self.baseline, self.trees = list(names), float(baseline), []
        total_nodes = 0
        for tree in trees:
            if not isinstance(tree, dict) or set(tree) != TREE_FIELDS:
                raise ValueError('Invalid classifier tree fields')
            if any(not isinstance(value, list) for value in tree.values()):
                raise ValueError('Classifier nodes must be arrays')
            size = len(tree['value'])
            total_nodes += size
            if not size or total_nodes > 500000 or any(len(value) != size for value in tree.values()):
                raise ValueError('Invalid classifier node counts')
            for key in ('feature', 'left', 'right'):
                if any(type(value) is not int for value in tree[key]):
                    raise ValueError('Classifier indices must be integers')
                limit = len(names) if key == 'feature' else size
                if any(not 0 <= value < limit for value in tree[key]):
                    raise ValueError('Classifier node index is out of bounds')
            for key in ('leaf', 'missing_left'):
                if any(type(value) is not bool for value in tree[key]):
                    raise ValueError('Classifier flags must be Boolean')
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in tree['value']):
                raise ValueError('Invalid classifier leaf values')
            threshold = np.array([_threshold(value) for value in tree['threshold']], dtype=np.float64)
            pending, visited = [0], set()
            while pending:
                node = pending.pop()
                if node < 0 or node >= size or node in visited:
                    raise ValueError('Classifier tree has a cycle, reused child or invalid index')
                visited.add(node)
                if not tree['leaf'][node]:
                    if not 0 <= tree['feature'][node] < len(names):
                        raise ValueError('Classifier feature index is outside its schema')
                    pending.extend((tree['left'][node], tree['right'][node]))
            if len(visited) != size:
                raise ValueError('Classifier contains unreachable nodes')
            converted = {key: np.asarray(value) for key, value in tree.items() if key != 'threshold'}
            converted['threshold'] = threshold
            self.trees.append(converted)
        self.boundary_refiner = None
        if document['format'] == BOUNDARY_FORMAT:
            from .boundary_runtime import BoundaryRefiner
            self.boundary_refiner = BoundaryRefiner(document.get('boundary_refiner'), self.names)
        elif 'boundary_refiner' in document:
            raise ValueError('Boundary refinement requires a version-four classifier')

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.names) or np.isinf(X).any():
            raise ValueError('Classifier input shape or values do not match its schema')
        raw = np.full(len(X), self.baseline, dtype=np.float64)
        for tree in self.trees:
            indices = np.zeros(len(X), dtype=np.int64)
            for _ in range(len(tree['value'])):
                active = np.flatnonzero(~tree['leaf'][indices])
                if not len(active):
                    break
                nodes = indices[active]
                values = X[active, tree['feature'][nodes]]
                left = np.where(np.isnan(values), tree['missing_left'][nodes],
                                values <= tree['threshold'][nodes])
                indices[active] = np.where(left, tree['left'][nodes], tree['right'][nodes])
            raw += tree['value'][indices]
        positive = raw >= 0
        probability = np.empty(len(raw), dtype=np.float64)
        probability[positive] = 1 / (1 + np.exp(-raw[positive]))
        exponential = np.exp(raw[~positive])
        probability[~positive] = exponential / (1 + exponential)
        return np.column_stack((1 - probability, probability))


def load(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Exported classifier is too large')
    document = json.loads(path.read_text(encoding='utf-8'))
    classifier = Classifier(document)
    if document['format'] in (DYNAMIC_FORMAT,CONTEXT_FORMAT,BOUNDARY_FORMAT):
        settings=document.get('feature_settings')
        if (not isinstance(settings,dict) or set(settings)!=set(FEATURE_SETTINGS)
                or type(settings['field_axis']) is not bool
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for k,v in settings.items() if k!='field_axis')
                or not 0<settings['min_person_h']<settings['max_person_h']<=1
                or any(settings[k]>1 for k in ('field_shrink_top','horizon_y'))):
            raise ValueError('Invalid classifier feature settings')
    fps, offsets, excluded = document.get('sample_fps'), document.get('offsets'), document.get('exclude')
    if (type(document.get('n_features')) is not int or document['n_features'] != len(classifier.names)
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or not isinstance(offsets, list) or not offsets
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in offsets)
            or not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded)):
        raise ValueError('Invalid exported classifier sampling/feature metadata')
    return classifier, document


def export(path, classifier, names, cfg, offsets):
    if not np.array_equal(getattr(classifier, 'classes_', []), [0, 1]):
        raise ValueError('Only numeric binary HGB classifiers with classes [0,1] can be exported')
    if getattr(classifier, 'n_features_in_', None) != len(names) or not hasattr(classifier, '_predictors'):
        raise ValueError('Classifier is not a fitted supported HGB model')
    trees = []
    for iteration in classifier._predictors:
        if len(iteration) != 1:
            raise ValueError('Multiclass classifier export is unsupported')
        nodes = iteration[0].nodes
        if 'is_categorical' in nodes.dtype.names and nodes['is_categorical'].any():
            raise ValueError('Categorical HGB splits cannot use the numeric inference format')
        thresholds = [float(value) if np.isfinite(value) else ('+inf' if value > 0 else '-inf')
                      for value in nodes['num_threshold']]
        trees.append(dict(feature=nodes['feature_idx'].astype(int).tolist(), threshold=thresholds,
                          left=nodes['left'].astype(int).tolist(), right=nodes['right'].astype(int).tolist(),
                          value=nodes['value'].astype(float).tolist(),
                          missing_left=nodes['missing_go_to_left'].astype(bool).tolist(),
                          leaf=nodes['is_leaf'].astype(bool).tolist()))
    pipeline=getattr(cfg,'feature_pipeline','legacy')
    if pipeline not in ('legacy','dynamic-field-v1'):raise ValueError('Unsupported classifier feature pipeline')
    if tuple(offsets) != tuple(cfg.model_offsets):raise ValueError('Export temporal offsets differ from configuration')
    context = cfg.model_feature_set != BASE or tuple(offsets) != DEFAULT_OFFSETS
    if context and pipeline != 'dynamic-field-v1':raise ValueError('Custom classifier features require the dynamic field profile')
    validate_offsets(offsets)
    document = dict(format=FORMAT if pipeline=='legacy' else (CONTEXT_FORMAT if context else DYNAMIC_FORMAT), names=list(names), baseline=float(classifier._baseline_prediction[0, 0]),
                    trees=trees, sample_fps=cfg.sample_fps, offsets=list(offsets),
                    exclude=list(cfg.model_exclude or ()), n_features=len(names))
    if pipeline!='legacy':
        document['feature_pipeline']=pipeline
        document['feature_settings']={key:getattr(cfg,key) for key in FEATURE_SETTINGS}
    if context:document['feature_set']=cfg.model_feature_set
    runtime = Classifier(document)
    probes = np.random.default_rng(0).normal(size=(256, len(names)))
    probes[::7, ::3] = np.nan
    if not np.allclose(runtime.predict_proba(probes), classifier.predict_proba(probes), rtol=0, atol=1e-12):
        raise ValueError('Exported classifier did not reproduce reference probabilities')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(document, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
