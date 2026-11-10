#!/usr/bin/env python3
"""Compare saved interval predictions; never train or select production models."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def candidates(report):
    if 'candidates' in report:
        return report['candidates']
    return [dict(variant=report['architecture'], confirmation=report['confirmation_cpu'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Requires optional matplotlib')
    args = parser.parse_args()
    path = Path(__file__).with_name('report-classifier-benchmark.py')
    spec = importlib.util.spec_from_file_location('boundary_report', path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    records = {item['name']: item for item in json.loads(manifest_bytes)}
    rows, sources, game_sets = [], {}, []
    for path in args.report:
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get('manifest_sha256', manifest_hash) != manifest_hash:
            raise ValueError(f'Ground truth manifest mismatch: {path}')
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        for index, candidate in enumerate(candidates(report)):
            result = candidate['confirmation']
            games = result['per_game']
            names = [game['game'] for game in games]
            if len(names) != len(set(names)) or not games:
                raise ValueError('Require nonempty, unique confirmation games')
            game_sets.append(tuple(sorted(names)))
            boundary = metrics.boundary_diagnostics([
                (records[game['game']]['marks'], game['predictions']) for game in games])
            matched = boundary['matched_plays']
            predicted = boundary['predicted_plays']
            per_game = [metrics.boundary_diagnostics([
                (records[game['game']]['marks'], game['predictions'])]) for game in games]
            label = candidate.get('variant', candidate.get('id', str(candidate.get('parameters', index))))
            rows.append(dict(source=str(path.resolve()), approach=label,
                precision=matched / predicted if predicted else 0.,
                recall=matched / boundary['ground_truth_plays'],
                minimum_game_recall=min(item['matched_plays'] / item['ground_truth_plays'] for item in per_game),
                **boundary))
    if len({row['ground_truth_plays'] for row in rows}) != 1:
        raise ValueError('Reports must cover the same GT play count')
    # Counts alone cannot establish equal game membership.
    if len(set(game_sets)) != 1:
        raise ValueError('Reports must cover identical games')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema=1, manifest_sha256=manifest_hash, source_sha256=sources,
        target=dict(event_precision=.95, event_recall=.95, both_ends_within_one_second_recall=.95),
        scope='Repeated confirmation diagnostics, not a fresh blind test or production selection.',
        candidates=rows)
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    columns = ['source', 'approach', 'precision', 'recall', 'minimum_game_recall',
        'both_boundaries_accurate_recall', 'both_boundaries_accurate_precision',
        'continuous_boundary_quality_recall', 'continuous_protective_quality_recall',
        'mean_absolute_boundary_error_seconds', 'mean_protective_boundary_cost_seconds']
    with (args.out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        points = axes[0].scatter([100*r['recall'] for r in rows], [100*r['precision'] for r in rows],
            c=[100*r['both_boundaries_accurate_recall'] for r in rows], cmap='viridis', vmin=0, vmax=100)
        fig.colorbar(points, ax=axes[0], label='Both ends within ±1 s / all GT (%)')
        axes[0].set(xlabel='Play recall (%)', ylabel='Play precision (%)')
        axes[0].axvline(95, color='gray', linestyle='--')
        axes[0].axhline(95, color='gray', linestyle='--')
        axes[1].scatter([r['mean_protective_boundary_cost_seconds'] for r in rows],
                        [100*r['both_boundaries_accurate_recall'] for r in rows], alpha=.6)
        axes[1].axhline(95, color='gray', linestyle='--', label='95% timing target')
        axes[1].set(xlabel='Matched-play boundary cost (s; cut-in penalty ×4)',
                    ylabel='Both ends within ±1 s / all GT (%)')
        axes[1].legend()
        fig.suptitle('Saved confirmation predictions — repeated validation, not a blind test')
        fig.savefig(args.out / 'comparison.png', dpi=170)
        plt.close(fig)
    print(f'Compared {len(rows)} saved approaches in {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
