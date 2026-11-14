# Complete play pipelines

The selectable unit is a **complete pipeline**: ordered source clips and a
configuration go in; global-time play intervals come out. Studio and CLI share
`create_pipeline(path).predict(clips, cfg) -> PipelineResult`. The result contains
plays, optional timestamp probabilities, and timing metadata. No manifest selects
arbitrary Python imports or downloads weights during inference.

The existing accepted pipeline remains the default. Three experimental recipes
can be imported separately: person features plus a temporal transformer, S3D video
features plus a temporal head, and MobileNet frame features plus a temporal head.
See [the comparison](research-ledger.md) before choosing them.
They have **not** earned promotion over the default.

## Studio and CLI

Use **Models → Import complete play pipeline…**, select its `pipeline.json`, then
choose it under the detection profile. Importing does not change the current
profile. Each choice shows its evaluation note in a tooltip. Whole-scene video
pipelines do not require field polygons; person pipelines retain field setup.
Selections and manifest checksums are snapshotted when queued. A changed or missing
recipe fails before launching the worker. Work remains in the cancellable worker
process, outside the Qt UI thread.

```bash
python -m scout play-pipeline import --file /private/recipe/pipeline.json
python -m scout play-pipeline list
python -m scout --source-dir /recordings/game \
  --play-pipeline /installed/recipe/pipeline.json run
```

The usual protection against replacing an existing review still applies. To
benchmark on real footage without writing to its Scout state, choose a **new**
experiment directory:

```bash
python -m scout --source-dir /recordings/game \
  --play-pipeline /installed/recipe/pipeline.json \
  benchmark-pipeline --output /experiments/new-run \
  --device cpu --threads 4 --source-index 0 --offset 480 --seconds 180
```

Use `--model-pack scout-reviewed-precision@0.5.0` instead of `--play-pipeline` to
measure the accepted baseline. Benchmarks include model loading, bounded decoding,
features, interval generation and review writing, with fresh private caches. They
exclude interpreter imports and MP4 encoding; measure external command wall time
as well when comparing short runs. Reference-frame seeks stay at their original
source timestamps. Original source size/mtime is checked before and after.

## Extending the implementation

`PlayDetectionPipeline` is the application boundary. `BaselinePipeline` adapts the
existing person/feature/classifier path; `ArtifactPipeline` binds encoder, named
features, temporal scorer and postprocessing into a single recipe. A new complete
implementation can implement the same `predict(clips, cfg)` protocol.

Within an artifact, `register_approach(ApproachInfo(...), scorer_factory,
provider=...)` adds a feature provider and scorer without modifying the UI or job
queue. A provider receives `(clips, cfg, spec, manifest_path)` and returns
`(Observations, timings)`; its scorer implements `predict(Observations) -> PlayScores`.
The provider must preserve clip ordering, use the global clip clock, release
subprocesses on failure/cancellation, and leave original media/state untouched.
The included built-ins use numeric ONNX heads. Registration is explicit code,
not executable instructions from an imported manifest.

For interchangeable person detectors, `scout.detector_adapters` provides a separate
box-level contract: BGR input batches, one normalized prediction per frame, pixel
`xyxy` boxes and confidence arrays. Named adapters have revisions in cache identity;
ordinary default cache keys stay compatible. This layer is optional for a video
pipeline.

A `scout-play-pipeline-v1` recipe binds feature order, context offsets, sampler,
feature precision/transform, encoder revision and **all** interval settings. Person
recipes require an exact installed detector pack. Local assets carry SHA-256
identities; imports validate and copy only declared inference files into immutable
version folders. Training data, GT and optimizer states are excluded. The ONNX head
includes the training-only learned normalizer. Visual `game-zscore` is an explicitly
bound unlabeled whole-game transform and requires offline full-game observations.

Visual encoders decode at 5 Hz and keep a bounded source-local frame ring and one
inference batch. Output features scale with game duration; decoded video does not.
Centered temporal windows repeat edges within their source. CPU uses supplied
ONNX assets, CUDA uses supplied tensor weights. Automatic GPU failure retries the
whole visual game on CPU, preventing mixed-backend features; explicit GPU failure
is reported. No Windows acceleration claim follows from Linux checks.

## Repeating research

Keep new outputs outside original Scout folders. The tools require new directories:

1. `scripts/extract-approach-features.py` reads a private frozen source manifest
   and supplied local encoder weights, producing reusable float16 features.
2. `scripts/compare-play-approaches.py` trains game-grouped temporal heads,
   calibrates on development out-of-fold probabilities, freezes selection, then
   scores the held-out games. `--help` exposes widths/epochs, visual sampling,
   transforms and calibration/reuse controls. Person sampling cannot be thinned
   independently because it changes tracking and derived features.
3. `scripts/build-play-pipelines.py` exports portable CPU encoders, verifies numeric
   parity, and builds one recipe per approach using development results only.
4. `benchmark-pipeline` measures the complete runtime through the same interface
   used by Studio. Do not compare contended shared extraction stage timings to
   sequential standalone measurements.

Cached features plus GT/split/schema suffice to retrain temporal heads. New visual
encoders need original video or an appropriate frame cache. Complete speed checks
need real video, and full-game validation needs the exact source sequence and GT.
There is no requirement to duplicate every MP4 into an experiment folder.
