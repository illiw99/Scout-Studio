#!/usr/bin/env python3
"""Read-only, game-grouped classifier comparison on prepared person features.

Input: an isolated approach-study manifest.json and person/<game>.npz (t, X,
names). X contains expanded, excluded features before temporal window stacking.
Never opens footage, reviews, production caches or pickles. Outputs are research
evidence, not automatically installable models. See docs/classifier-options-2026-11-10.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scout import model
from scout.config import Config
from scout.model_evaluation import interval_metrics, summarize
from scout.storage import validate_name

OFFSETS = (-10., -6., -3., -1., 0., 1., 3., 6., 10.)
VARIANTS = ('hgb-reference', 'hgb-small', 'mlp32', 'mlp64-32',
            'adaboost-stumps', 'adaboost-depth2', 'extra-trees-small',
            'extra-trees-deep', 'random-forest', 'logistic')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def make_classifier(name, seed, threads):
    from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                                  RandomForestClassifier, HistGradientBoostingClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    if name.startswith('hgb-'):
        reference = name == 'hgb-reference'
        return HistGradientBoostingClassifier(max_depth=5 if reference else 3,
            max_iter=250 if reference else 150, min_samples_leaf=60,
            learning_rate=.06, l2_regularization=1. if reference else 3.,
            early_stopping=False, random_state=seed)
    if name.startswith('mlp'):
        estimator = MLPClassifier(hidden_layer_sizes=(32,) if name == 'mlp32' else (64, 32),
            alpha=.1, max_iter=80, batch_size=256, early_stopping=False,
            random_state=seed, solver='adam')
    elif name == 'logistic':
        estimator = LogisticRegression(C=.1, max_iter=500, random_state=seed)
    elif name.startswith('adaboost-'):
        estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(
            max_depth=1 if name.endswith('stumps') else 2, min_samples_leaf=30,
            random_state=seed), n_estimators=100, learning_rate=.5, random_state=seed)
    elif name.startswith('extra-trees'):
        estimator = ExtraTreesClassifier(n_estimators=100,
            max_depth=8 if name.endswith('small') else 16, min_samples_leaf=10,
            n_jobs=threads, random_state=seed)
    elif name == 'random-forest':
        estimator = RandomForestClassifier(n_estimators=100, max_depth=12,
            min_samples_leaf=10, n_jobs=threads, random_state=seed)
    else:
        raise ValueError('Unknown classifier')
    # Fit imputation/scaling only on the fold's training games. Keep all-missing
    # columns so the exact temporal schema remains stable across folds.
    steps = [SimpleImputer(strategy='median', keep_empty_features=True)]
    if name.startswith('mlp') or name == 'logistic':
        steps.append(StandardScaler())
    return make_pipeline(*steps, estimator)


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty game manifest')
    names = set()
    for record in records:
        validate_name(record['name'])
        if record['name'] in names or record['partition'] not in ('development', 'holdout', 'confirmation'):
            raise ValueError('Duplicate games or invalid partition')
        names.add(record['name'])
        if not isinstance(record['group'], str) or not record['group']:
            raise ValueError('Missing match-day group')
        marks = np.asarray(record['marks'], dtype=float).reshape(-1, 2)
        if not len(marks) or not np.isfinite(marks).all() or np.any(marks[:, 1] <= marks[:, 0]):
            raise ValueError('Invalid ground truth')
    development = {r['group'] for r in records if r['partition'] == 'development'}
    holdout = {r['group'] for r in records if r['partition'] == 'holdout'}
    if len(development) < 2 or not holdout or development & holdout:
        raise ValueError('Require independent development folds and held-out days')
    return sorted(development)


def load_case(study, record, expected_names, fps):
    path = study / 'person' / (record['name'] + '.npz')
    with np.load(path, allow_pickle=False) as archive:
        t, X, names = archive['t'], archive['X'], archive['names'].tolist()
    if (names != expected_names or t.ndim != 1 or not len(t)
            or X.shape != (len(t), len(names)) or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or np.isinf(X).any()):
        raise ValueError('Invalid or incompatible prepared features')
    # Features may contain NaN (e.g. no players); each model handles them using
    # its training-only pipeline. Temporal edge convention matches Scout.
    X = model.window(X, fps, OFFSETS).astype(np.float32)
    truth = np.asarray(record['marks'], dtype=float)
    y = np.zeros(len(t), dtype=np.int8)
    for start, end in truth:
        y[(t >= start) & (t <= end)] = 1
    if len(np.unique(y)) != 2:
        raise ValueError('Every game needs both play and non-play frames')
    return dict(record=record, t=t, X=X, y=y, sha256=digest(path))


def score(cases, probabilities, cfg):
    events, strict, games = [], [], []
    for case, prob in zip(cases, probabilities):
        if prob.shape != case['t'].shape or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
            raise ValueError('Invalid classifier probabilities')
        record = case['record']
        end = record['duration']
        spans = [(max(0., a), min(end, b)) for a, b in model.postprocess(cfg, case['t'], prob)
                 if min(end, b) > max(0., a)]
        result = interval_metrics(record['marks'], spans, 0)
        events.append(result)
        strict.append(interval_metrics(record['marks'], spans, .5))
        summary = summarize([result])
        # Missed GT plays count as boundary failures in this denominator.
        boundary_recall = {str(t): sum(abs(a) <= t and abs(b) <= t for a, b in
            zip(result['start_errors'], result['end_errors'])) / len(record['marks'])
            for t in (.5, 1., 2.)}
        games.append(dict(game=record['name'], predictions=spans,
            boundary_recall=boundary_recall, **summary))
    summary = summarize(events)
    start = [x for result in events for x in result['start_errors']]
    end_errors = [x for result in events for x in result['end_errors']]
    total = summary['hits'] + summary['missed']
    boundary = {str(t): sum(abs(a) <= t and abs(b) <= t for a, b in zip(start, end_errors)) / total
                for t in (.5, 1., 2.)}
    return dict(event=summary, iou50=summarize(strict), boundary_recall=boundary,
        start_p90=float(np.percentile(np.abs(start), 90)) if start else None,
        end_p90=float(np.percentile(np.abs(end_errors), 90)) if end_errors else None,
        minimum_game_recall=min(game['recall'] for game in games), per_game=games)


def rank(result, minimum_recall):
    good = result['minimum_game_recall'] >= minimum_recall
    # Precision first under recall constraint; reward useful timestamps next.
    return (good, result['event']['precision'] if good else result['minimum_game_recall'],
            result['boundary_recall']['1.0'], result['iou50']['recall'],
            -(result['event']['boundary_p90'] if result['event']['boundary_p90'] is not None else 1e30))


def tune(cases, probabilities, cfg, minimum_recall):
    pooled = np.concatenate(probabilities)
    thresholds = sorted(set([.01, .05, .1, .2, .35, .5, .65, .8, .9, .95, .99,
                            *map(float, np.quantile(pooled, [.5, .7, .8, .9, .95, .98, .99]))]))
    trials = []
    for threshold in thresholds:
        for duration in (2., 4.):
            params = dict(model_threshold=threshold, model_min_duration=duration,
                          model_smooth=.2, model_peak_threshold=0., model_refine_fraction=0.)
            trials.append(dict(parameters=params, **score(cases, probabilities, replace(cfg, **params))))
    seeds = sorted(trials, key=lambda x: rank(x, minimum_recall), reverse=True)[:3]
    for seed in seeds:
        for smooth in (.2, .6):
            for peak in (0., .6):
                for refine in (0., .5):
                    params = dict(seed['parameters'], model_smooth=smooth,
                                  model_peak_threshold=peak, model_refine_fraction=refine)
                    trials.append(dict(parameters=params, **score(cases, probabilities, replace(cfg, **params))))
    return max(trials, key=lambda x: rank(x, minimum_recall)), len(trials)


def fit(classifier, cases):
    X = np.concatenate([case['X'] for case in cases])
    y = np.concatenate([case['y'] for case in cases])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        classifier.fit(X, y)
    return time.perf_counter() - started, sorted({str(item.message) for item in captured})


def run(args):
    from threadpoolctl import threadpool_limits
    study, output = args.study.resolve(), args.out.resolve()
    if output == study or study in output.parents:
        raise ValueError('Choose a new output directory outside the input study')
    if output.exists():
        raise ValueError('Benchmark outputs are never overwritten')
    manifest = study / 'manifest.json'
    records = json.loads(manifest.read_text())
    groups = validate_records(records)
    # The baseline locks the prepared schema and common temporal inputs.
    baseline_file = study / 'baseline.json'
    baseline = json.loads(baseline_file.read_text())
    cfg = Config(**baseline['configuration'])
    if tuple(cfg.model_offsets) != OFFSETS:
        raise ValueError('Unexpected baseline temporal offsets')
    fps = cfg.sample_fps
    first = next(record for record in records if record['partition'] == 'development')
    with np.load(study / 'person' / (first['name'] + '.npz'), allow_pickle=False) as archive:
        names = archive['names'].tolist()
    if model.window_names(names, OFFSETS) != baseline['classifier']['names']:
        raise ValueError('Prepared feature schema differs from baseline')
    development = [load_case(study, record, names, fps) for record in records if record['partition'] == 'development']
    output.mkdir(parents=True)
    report = dict(schema=1, scope='classifier-only research; repeated held-out confirmation, not a blind test',
        configuration=asdict(cfg), offsets=OFFSETS, threads=args.threads, seed=args.seed,
        minimum_game_recall=args.minimum_recall,
        selection='development-only event precision under per-game recall floor; boundary recall and IoU tie breaks',
        manifest_sha256=digest(manifest), baseline_sha256=digest(baseline_file),
        versions={name: version(name) for name in ('numpy', 'scikit-learn', 'scipy')},
        code_sha256={name: digest(Path(__file__).resolve().parents[1] / name) for name in
            ('scripts/benchmark-classifiers.py', 'scout/model.py', 'scout/model_evaluation.py')},
        feature_sha256={case['record']['name']: case['sha256'] for case in development},
        development_games=[case['record']['name'] for case in development], candidates=[])
    with threadpool_limits(limits=args.threads):
        for variant in args.variants:
            probabilities = {}
            fold_reports = []
            for group in groups:
                training = [case for case in development if case['record']['group'] != group]
                validation = [case for case in development if case['record']['group'] == group]
                classifier = make_classifier(variant, args.seed, args.threads)
                elapsed, messages = fit(classifier, training)
                for case in validation:
                    probabilities[case['record']['name']] = classifier.predict_proba(case['X'])[:, 1]
                fold_reports.append(dict(group=group, training_games=[case['record']['name'] for case in training],
                    validation_games=[case['record']['name'] for case in validation], fit_seconds=elapsed, warnings=messages))
                print(json.dumps(dict(variant=variant, fold=group, fit_seconds=elapsed)), flush=True)
            ordered = [probabilities[case['record']['name']] for case in development]
            best, trials = tune(development, ordered, cfg, args.minimum_recall)
            folder = output / variant
            folder.mkdir()
            for case, prob in zip(development, ordered):
                np.savez_compressed(folder / (case['record']['name'] + '-oof.npz'), t=case['t'], prob=prob)
            result = dict(variant=variant, folds=fold_reports, development=best, tuning_trials=trials)
            report['candidates'].append(result)
            write_new(folder / 'development.json', result)
            print(json.dumps(dict(variant=variant, development=best['event'], minimum_game_recall=best['minimum_game_recall'])), flush=True)
        # Freeze ALL family parameters and the overall choice before loading any
        # held-out feature array. No validation result can change this selection.
        selected = max(report['candidates'], key=lambda x: rank(x['development'], args.minimum_recall))
        report['selected_on_development'] = selected['variant']
        write_new(output / 'selection.json', report)
        confirmation = [load_case(study, record, names, fps) for record in records if record['partition'] != 'development']
        for case in confirmation:
            report['feature_sha256'][case['record']['name']] = case['sha256']
        for result in report['candidates']:
            variant = result['variant']
            classifier = make_classifier(variant, args.seed, args.threads)
            elapsed, messages = fit(classifier, development)
            probabilities, timings = [], []
            for case in confirmation:
                classifier.predict_proba(case['X'][:min(256, len(case['X']))])  # untimed warmup
                repeats = []
                for _ in range(3):
                    start = time.perf_counter()
                    prob = classifier.predict_proba(case['X'])[:, 1]
                    repeats.append(time.perf_counter() - start)
                probabilities.append(prob)
                timings.append(dict(game=case['record']['name'], rows=len(prob), prediction_seconds=repeats,
                    median_seconds=float(np.median(repeats)), video_seconds=case['record']['duration']))
                np.savez_compressed(output / variant / (case['record']['name'] + '-confirmation.npz'), t=case['t'], prob=prob)
            operating = replace(cfg, **result['development']['parameters'])
            result.update(confirmation=score(confirmation, probabilities, operating), timing=timings,
                final_fit_seconds=elapsed, final_fit_warnings=messages,
                prediction_scope='prepared windowed features to probabilities only; excludes decode, detector, features and postprocessing')
            write_new(output / variant / 'result.json', result)
            print(json.dumps(dict(variant=variant, confirmation=result['confirmation']['event'])), flush=True)
    # Verify all read-only input bytes again. No source video is ever opened.
    for name, expected in report['feature_sha256'].items():
        if digest(study / 'person' / (name + '.npz')) != expected:
            raise ValueError('Input features changed during benchmark')
    if digest(manifest) != report['manifest_sha256'] or digest(baseline_file) != report['baseline_sha256']:
        raise ValueError('Input manifest/baseline changed during benchmark')
    report['inputs_unchanged'] = True
    write_new(output / 'report.json', report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--minimum-recall', type=float, default=.95)
    args = parser.parse_args()
    if not 0 < args.minimum_recall <= 1 or args.threads < 1 or len(set(args.variants)) != len(args.variants):
        parser.error('Invalid recall, thread count or duplicate variants')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
