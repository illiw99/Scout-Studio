# Model evaluation and improvement protocol

Current results: [model status](model-status.md). Completed experiments and
negative results: [research ledger](research-ledger.md). This page defines the
protocol; it is not a chronological work log.

## Goals and data

Precision is primary; aim 95–98% event precision/recall with at least 95% recall in
every measured game. Aim 95% of all GT plays with both cuts within ±1s; misses fail.
Early starts/late ends are preferable to cut-in, but overlong padding is not a
substitute for timing accuracy. CPU 5–10× / GPU ≈15× are complete-pipeline goals.
Use no audio-assisted features. Reviewed GT is useful but imperfect; it labels
play intervals, not person bounding boxes.

The established split has five development games / 405 plays (GD1 TIG–THU,
W2 CON/VET, W3 DUC, GD3 INV), four confirmation games / 328 plays (GD4 CON/RAI,
W3 HON/PAN). Hold out whole match days inside development; W2 games share a fold.
HON/PAN share W3 with DUC, and confirmation has been inspected repeatedly:
this is not a blind unseen-day test. Historical NT17 measurements exist, but its
videos are now placeholders and cannot establish fresh media-backed validation.

## Before fitting

1. Snapshot authoritative current reviews, field polygons, source ordering,
   size/mtime identities and cache hashes read-only into a new isolated study.
   Preserve point markers, original files, models and state. Skip missing,
   changing or Nextcloud placeholder videos; do not silently hydrate them.
2. Record unverifiable legacy cache identities. Rebuild global timestamps from
   the reviewed source order; do not inherit stale cache timeline offsets.
3. Fit scalers, feature selection and classifiers on training folds only. Never
   randomly split neighboring frames. Nested upstream fits must exclude the
   outer validation day and the inner sample's own day.
4. Preserve exact source boundaries for sequence/pool windows; record any
   alignment tolerance. The source-join correction in the ledger is a required
   regression case for future pooled features.

## Selection, measurement and promotion

- Select architecture, thresholds and postprocessing using development OOF
  predictions; write the frozen selection before loading confirmation arrays.
- Compare identical games/GT with one-to-one positive-overlap matching and
  IoU≥.5. Report precision, recall, false/missed plays, per-game recall, joint
  boundary success over all GT, separate starts/ends, MAE/p90 and extra/lost
  action. [Boundary quality](boundary-quality.md) defines all costs/denominators.
- Recompute tracking/features when changing sample rate. A lower resolution,
  detector, quantization or decoder needs fresh detections and quality scoring;
  thinning old observations does not prove another detector's quality.
- Time startup, decode, detector, features, classifier, refinement and review
  writing end to end. Report downloads/exports, cold startup and MP4 encoding
  separately. Head-only latency is not realtime processing speed.
- Retain configurations, fold membership, seeds, dependency/source versions,
  hashes, predictions, failed attempts and candidate checkpoints. A compact
  rejection ledger prevents duplicate trials without discarding raw evidence.
- Before promotion, check numeric/ONNX parity, source-time conventions, cache
  identity, CPU fallback and standalone compatibility. New source/feature
  contracts need explicit validation. Keep rollback/manual choices intact.
- Require fresh independent game/day confirmation for a generalization claim.
  Repeatedly selecting on inspected confirmation is adaptive research, not a
  new held-out guarantee. Never promote merely because an architecture is newer.

## Tools

| Task | Entry point |
| --- | --- |
| Reviewed snapshots and interval scoring | `scripts/evaluate-reviewed-games.py`, `scout/model_evaluation.py` |
| Threshold calibration | `scripts/calibrate-play-thresholds.py` |
| Controlled feature extraction/fusion | `scripts/extract-video-features.py`, `scripts/combine-video-features.py`, `scripts/prepare-camera-region-features.py` |
| Freeze and verify decision candidate | `scripts/freeze-reviewed-candidate.py`, `scripts/evaluate-frozen-candidate.py` |
| Classifier comparison | [classifier benchmark](classifier-benchmark.md) |
| Full approach comparison/recipes | `scripts/compare-play-approaches.py`, `scripts/build-play-pipelines.py` |
| Boundary comparison/plots | `scripts/report-classifier-benchmark.py`, `scripts/compare-boundary-approaches.py` |
| Complete runtime benchmarks | `scripts/benchmark-reviewed-pipeline.py`, `scripts/benchmark-boundary-pipeline.py` |

All experimental outputs belong outside original game folders. No GT,
footage, caches or private experiment environments belong in release packages.
