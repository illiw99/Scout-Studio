# Current model and measured results

Status: 14 November 2026. This page is the canonical release/model summary.

Scout Studio 1.0.3 packages the frozen visual boundary-window candidate as
`scout-reviewed-precision@0.5.0`. The prior 0.4.0 pack remains immutable and is
available for rollback and the Person Transformer's detector/feature dependency.
Explicit user selections remain preserved during automatic default updates.

## What is saved

The person detector, tracking, 31 derived features, nine temporal offsets and
base HGB decision trees are unchanged. Two trained binary HGB heads score
candidate start/end positions within six seconds in 0.4-second steps. Inputs
include local feature values, changes, curvature, probabilities, duration and
candidate shift. A deterministic tie preference keeps smaller corrections.
Conflicting corrections revert to the original intervals so plays cannot be
reordered or acquire new overlaps. Existing explicit recipe overlaps are retained.

The export is numeric JSON (`scout-hgb-v4`), approximately 800 KB including the
base head. Application inference uses NumPy only, with no pickle, training
framework, audio features or additional vision backbone. Older engines reject
the new format instead of silently skipping refinement. The research checkpoint
SHA-256 is `e8e1555afec9506ab0259a13e35f322f4bed54b863e8724970a7e644bdca516c`.

## Quality and limitations

Training used five development games (405 GT plays), with nested whole-day
upstream fitting. Configuration was selected on development predictions before
confirmation. The four confirmation games contain 328 GT plays and have been
inspected repeatedly; they are not a fresh blind test. Initial legacy detector
cache identities were not fully authenticated. These results are research
evidence, not a universal accuracy guarantee or fresh CPU-detector quality test.

| Metric | Prior HGB | Packaged visual refiner |
| --- | ---: | ---: |
| Play precision | 92.05% | 92.05% |
| Play recall | 98.78% | 98.78% |
| Minimum game recall | 96.15% | 96.15% |
| Both cuts within ±1 second / all GT | 67.99% | 73.48% |
| Start within ±1 second / all GT | 90.55% | 95.43% |
| End within ±1 second / all GT | 73.48% | 76.22% |

The packaged NumPy runtime reproduces all saved confirmation intervals, including
the conflict guard. Missed plays fail joint timing accuracy. The 95% joint timing
and 95–98% precision/recall aims remain unmet; ending accuracy is the main gap.
See [metric definitions](boundary-quality.md). Export pre/post padding remains
an independent user setting and does not change these raw detection metrics.

## Speed measurement

The numeric refiner alone measured approximately 0.044–0.053 seconds per complete
confirmation game on prepared features (four CPU threads). This excludes decode,
detection, tracking and the base classifier.

`scripts/benchmark-boundary-pipeline.py` measures the complete processing path on
an explicit bounded source window, using fresh private detection caches for each
repeat. It includes decode, detection, adaptive fields, tracking, features, the
base classifier and refinement. One-time model download/export and MP4 export
are excluded and preparation time is recorded separately. Original footage and
field JSON hashes are checked unchanged. No generated state goes into game folders.

On this Intel i7-11700F / RTX 3060 Ti system, the 120-second active-play window
at source time 499 seconds in the development game measured **6.43× realtime on
CPU**, median of three runs. CPU configuration: OpenVINO, 1280 input size,
2.5 sampled frames/s, eight threads. The refiner cost about six milliseconds for
the five proposals. This is a bounded source-build measurement, not a full-game,
Windows or frozen-installer throughput guarantee. The identical window measured **13.95× realtime on CUDA**, median of three runs,
with batch16. Detection/decode alone was approximately16×; field adaptation and
other stages are included in the13.95× whole-pipeline figure. The15× GPU aim is
not yet established end-to-end on this bounded window. Source dependencies were
Torch2.14.1, Ultralytics8.4.183 and OpenVINO2026.4.1; these timings do not establish
performance of a different pinned installer environment.

Example (use a new workspace outside original game folders):

```bash
python scripts/benchmark-boundary-pipeline.py --source /path/to/source.MP4 \
  --workspace /path/to/new-benchmark --start 499 --duration 120 \
  --device cpu --threads 8 --repeats 3
```

The detector and inference settings must remain bound to this model pack. Bigger
feature pools are a subsequent experiment: retain only features that improve
game-held-out quality at acceptable CPU cost. Player-interaction, sudden-stop
and cluster-transition cues are candidates; no gain from them is claimed here.

## Published release verification — 14 November 2026

[Studio 1.0.3](https://github.com/illiw99/Scout-Studio/releases/tag/v1.0.3) is
published from `ce53c48`. All four native builds and installed/upgrade smoke
checks passed in [Actions](https://github.com/illiw99/Scout-Studio/actions/runs/43484622175),
and both source-test jobs passed. Promotion passed without rebuilding the
installers. The live GitHub API exposes exactly four installer assets: Linux
deb (209.3MB), Windows exe (138.2MB), Mac ARM dmg (150.1MB), Mac Intel dmg
(189.9MB), decimal sizes. No runtime parts or support releases are published.

All four GitHub upload digests match the tested installer identities embedded
in release notes, and the updater accepted that live metadata for every target.
CPU/CUDA/native processing packages install from pinned upstream sources into
private Python environments. Existing Studio 1.0.2 and earlier need a one-time
manual download of the new installer; their old updater cannot read this new
metadata format. Actual NVIDIA performance on a physical Windows machine
remains unverified by GPU-less hosted tests.

The model research continued separately with
[feature pools and compact neural heads](research-ledger.md).
Those candidates do not replace the frozen release model.

For rejected approaches, corrected historical results and experiment paths, use
the [research ledger](research-ledger.md). Evaluation rules live in
[model improvement](model-improvement.md); unfinished work is in the [roadmap](future-path.md).
