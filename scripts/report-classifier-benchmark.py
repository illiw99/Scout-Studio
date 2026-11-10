#!/usr/bin/env python3
"""Summarize frozen classifier results with asymmetric play-boundary diagnostics.

No fitting, probability calibration, model selection or footage access occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scout.model_evaluation import interval_metrics


def boundary_diagnostics(truth_and_predictions, tolerance=1., cut_weight=4.):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive timing tolerance required')
    if not np.isfinite(cut_weight) or cut_weight < 1:
        raise ValueError('Cut weight must be finite and at least one')
    results = [interval_metrics(truth, predictions, 0) for truth, predictions in truth_and_predictions]
    start = np.asarray([value for result in results for value in result['start_errors']])
    end = np.asarray([value for result in results for value in result['end_errors']])
    total = sum(result['hits'] + result['missed'] for result in results)
    if not total:
        raise ValueError('Expected annotated plays')
    late_start, early_end = np.maximum(start, 0), np.maximum(-end, 0)
    early_start, late_end = np.maximum(-start, 0), np.maximum(end, 0)
    predictions = sum(result['hits'] + result['false_positives'] for result in results)
    accurate = int(np.count_nonzero((np.abs(start) <= tolerance) & (np.abs(end) <= tolerance)))
    absolute = np.abs(start) + np.abs(end)
    weighted = cut_weight * (late_start + early_end) + early_start + late_end
    def summary(errors):
        return dict(affected_plays=int(np.count_nonzero(errors > 1e-6)),
                    mean_seconds=float(np.mean(errors)) if len(errors) else None,
                    p90_seconds=float(np.percentile(errors, 90)) if len(errors) else None)
    return dict(tolerance_seconds=tolerance, cut_weight=cut_weight,
        ground_truth_plays=total, predicted_plays=predictions, matched_plays=len(start),
        accurate_plays=accurate,
        start_within_tolerance_recall=float(np.count_nonzero(np.abs(start) <= tolerance) / total),
        end_within_tolerance_recall=float(np.count_nonzero(np.abs(end) <= tolerance) / total),
        both_boundaries_accurate_precision=accurate / predictions if predictions else 0.,
        continuous_boundary_quality_recall=float(np.exp(-absolute / (2 * tolerance)).sum() / total),
        continuous_protective_quality_recall=float(np.exp(-weighted / (2 * tolerance)).sum() / total),
        mean_absolute_boundary_error_seconds=float(absolute.mean() / 2) if len(start) else None,
        mean_protective_boundary_cost_seconds=float(weighted.mean() / 2) if len(start) else None,
        late_start=summary(late_start), early_end=summary(early_end),
        early_start=summary(early_start), late_end=summary(late_end),
        both_boundaries_accurate_recall=accurate / total,
        covers_play_within_tolerance_recall=float(np.count_nonzero(
            (start <= tolerance) & (end >= -tolerance)) / total),
        matched_plays_cut_by_more_than_tolerance=int(np.count_nonzero(
            (late_start > tolerance) | (early_end > tolerance))),
        mean_lost_action_seconds=float(np.mean(late_start + early_end)) if len(start) else None,
        mean_extra_context_seconds=float(np.mean(early_start + late_end)) if len(start) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=1.)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    manifest = args.manifest.read_bytes()
    if (report.get('inputs_unchanged') is not True
            or hashlib.sha256(manifest).hexdigest() != report['manifest_sha256']):
        raise ValueError('Require a completed benchmark and matching GT manifest')
    records = {record['name']: record for record in json.loads(manifest)}
    rows = []
    for candidate in report['candidates']:
        games = candidate['confirmation']['per_game']
        diagnostics = boundary_diagnostics([(records[game['game']]['marks'], game['predictions']) for game in games], args.tolerance)
        rows.append(dict(variant=candidate['variant'], event=candidate['confirmation']['event'],
            minimum_game_recall=candidate['confirmation']['minimum_game_recall'],
            iou50_recall=candidate['confirmation']['iou50']['recall'],
            boundary=diagnostics, prediction_seconds=sum(item['median_seconds'] for item in candidate['timing'])))
    result = dict(schema=1, benchmark_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        selected_on_development=report['selected_on_development'], candidates=rows,
        scope='Post-selection diagnostics only; held-out results do not change the frozen selection. Coverage tolerates extra context; inspect IoU and extra seconds to expose overlong clips.')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Classifier confirmation and boundary accuracy', '', result['scope'], '',
        f'Timing tolerance: {args.tolerance:g} s. Boundary recall denominators include missed GT plays.', '',
        '| Variant | Precision | Recall | Worst game recall | IoU≥0.5 recall | Both ends accurate | Covers play within tolerance | Lost action s/play | Extra context s/play | CPU head seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        event, boundary = row['event'], row['boundary']
        percentage = [event['precision'], event['recall'], row['minimum_game_recall'], row['iou50_recall'],
                      boundary['both_boundaries_accurate_recall'], boundary['covers_play_within_tolerance_recall']]
        values = ' | '.join(f'{value:.2%}' for value in percentage)
        lost, extra = boundary['mean_lost_action_seconds'], boundary['mean_extra_context_seconds']
        seconds = ' | '.join('—' if value is None else f'{value:.3f}' for value in (lost, extra, row['prediction_seconds']))
        lines.append(f"| {row['variant']} | {values} | {seconds} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
