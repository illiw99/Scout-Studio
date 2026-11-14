# Research ledger: results and decisions

Updated 14 November 2026. Start with [current model status](model-status.md).
This is the compact memory of completed experiments, including failures. A
rejected configuration is not proof that an entire architecture can never work.
**Do not repeat the same configuration/data/split without a specific new reason.**

## Common interpretation

Nine reviewed games, 733 GT plays: five development games/405 plays, four
confirmation games/328 plays. Development folds hold out entire match days;
HON/PAN confirmation shares W3 with development DUC. Confirmation has been
inspected repeatedly, so it is not blind. Initial cache identities were not
fully authenticated. Select parameters on development predictions before
confirmation, normalize on training folds only, and preserve source ordering.
All timings need a scope: head-only, detection-only and complete pipeline results
are not interchangeable. Detector box agreement is not labeled play recall.
See [protocol](model-improvement.md) and [timing metrics](boundary-quality.md).
No audio-assisted approach is eligible. Clip export audio is unaffected.

## Decisions: what not to repeat unchanged

| Approach | Recorded result / reason | Decision and condition for revisiting |
| --- | --- | --- |
| YOLO11m at 1280, 5 Hz, CPU batch 8 | Historical detection stage 0.54× on 18 s; medium detector dominates cost | Superseded by YOLO26n 1280 / 2.5 Hz. Revisit only with a new runtime/quality benefit |
| Lower resolution alone | At 960, YOLO11m retained 74.4% and YOLO11s 70.5% of their own 1280 field boxes on eight frames; visible distant-player loss | Require fresh game-level quality checks |
| Sampling 1 Hz | Development IoU≥0.5 F1 was 55.49%, versus 91.14% at 2.5 Hz in the early HGB screen | Do not adopt without redesigned tracking/sampling and recall evidence |
| Static fields / naive sparse tracker | Camera motion and frame-based gates damaged recall | Retain adaptive fields and time-scaled tracking |
| Person Transformer | P87.53/R98.48, versus HGB92.05/98.78 on 328 GT plays; more false plays | Keep selectable experiment; no promotion without precision gain |
| Frozen S3D/MobileNet person-free heads | S3D P40.24/R91.16; MobileNet P35.61/R58.84 | Retired binaries. Require a new encoder/training strategy |
| Motion-only / motion+S3D screening | Weakest-game recall 18.99–23.23% in early development | Reject as standalone replacements |
| Alternative sklearn classifiers | Full table below; no qualifying selected replacement | Do not rescan identical features/folds/hyperparameters |
| CatBoost / LightGBM / fixed probability ensembles | Did not beat HGB; selected ensemble minimum-game recall 94.87% | Reject unchanged; revisit with new evidence/features |
| Full ActionFormer/TriDet adaptations | Better timing, P81.45/82.52; protective TriDet P82.44/R98.78/joint75.00 | Too many false positives; need a validated verifier or new representation |
| Camera-compensated motion-grid refiner | Selected joint 70.73%; nine-game extraction 26.5 CPU minutes | No gain over the 73.48% release refiner; avoid added decode cost unchanged |
| S3D boundary context | Joint 69.82% | No promotion |
| Cadence, recording context, balanced proposal negatives | Selected settings did not improve confirmation P/R | Reject unchanged |
| Interaction cues + long activity pool | Selected joint 73.78%; eleven new cues did not beat simpler pool | No extra production stage |
| Six proposal verifiers, 21 thresholds each | Selected threshold kept every proposal; no precision gain under the 95% per-game development recall floor | Reject unchanged; no confirmation threshold cherry-picking |
| Eight end-only long-pool head settings | Selected joint 74.39%; no major gain from depth/tolerance/regularization/iterations | Retain the original small head as reference |
| Compact neural boundary heads | Selected soft-GRU joint 72.87%, below HGB73.48 | Not promoted; small screen, not exhaustive NN optimization |
| Protective margin calibration (14 November) | Pre0.2/post0.6 halves mean lost action, but joint timing falls73.48→68.60%; gentle0.2/0.2 gives72.87% | Safety/context tradeoff, not learned timing gain; [details](boundary-quality.md#protective-boundaries-versus-exact-timing) |
| Audio features | Explicitly disallowed; derived arrays/checkpoints removed | Do not retry without a changed user instruction |

## Classifier and full temporal-head comparison — 10 November

Same328 GT, selected development operating points. The HGB row here predates the
release boundary refiner: joint67.99% versus73.48% with that refiner. CPU times
cover prepared head inference, not the full pipeline. TAL timings additionally
include normalization/network/NMS, so they are stage costs, not identical kernels.
Full ActionFormer/TriDet heads were adapted to one play class and31 person features;
these are not reproductions of the papers' published benchmark scores.

| Variant | Precision | Recall | Worst-game recall | Both ends within 1s / all GT | Boundary p90 s | CPU head seconds, four games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgb-reference | 92.05% | 98.78% | 96.15% | 67.99% | 1.400 | 0.144 |
| hgb-small | 91.24% | 98.48% | 94.87% | 65.24% | 1.518 | 0.084 |
| mlp32 | 87.70% | 97.87% | 94.87% | 63.11% | 1.424 | 0.057 |
| mlp64-32 | 90.62% | 97.26% | 91.03% | 63.11% | 1.580 | 0.081 |
| adaboost-stumps | 87.99% | 96.04% | 93.10% | 0.00% | 12.342 | 0.369 |
| adaboost-depth2 | 89.01% | 98.78% | 97.44% | 0.00% | 7.209 | 0.439 |
| extra-trees-small | 64.23% | 96.34% | 94.25% | 3.66% | 11.150 | 0.134 |
| extra-trees-deep | 76.18% | 98.48% | 97.44% | 63.41% | 1.687 | 0.187 |
| random-forest | 79.27% | 99.09% | 98.72% | 65.24% | 1.463 | 0.151 |
| logistic | 87.98% | 98.17% | 94.87% | 41.46% | 2.600 | 0.066 |
| mlp16 | 90.34% | 96.95% | 94.87% | 59.45% | 1.599 | 0.062 |
| mlp128-64 | 90.72% | 95.43% | 88.46% | 57.32% | 1.605 | 0.313 |
| lstm16 | 91.43% | 97.56% | 94.87% | 67.68% | 1.332 | 0.084 |
| lstm32 | 91.69% | 97.56% | 94.87% | 65.85% | 1.436 | 0.122 |
| actionformer | 81.45% | 99.09% | 98.70% | 74.09% | 1.210 | 7.820 |
| tridet | 82.52% | 97.87% | 96.15% | 74.09% | 1.084 | 5.598 |
| tridet128 | 70.43% | 98.78% | 97.44% | 71.04% | 1.245 | 1.044 |
| tridet-s3d | 25.92% | 94.82% | 88.46% | 35.37% | 3.241 | 5.081 |
| tridet-mobile | 21.08% | 77.13% | 67.53% | 11.89% | 4.714 | 5.405 |
| tridet-protective-4x | 82.44% | 98.78% | 97.44% | 75.00% | 1.188 | 5.065 |

## Boundary candidates and corrections

All following rows use the same four confirmation games and retain event
P92.05/R98.78, minimum-game recall96.15%, unless stated otherwise. Misses fail
joint timing; export padding does not improve raw detection metrics.

| Candidate | Joint both cuts±1s/all GT | Status |
| --- | ---: | --- |
| Frozen HGB without boundary head |67.99%| Historical baseline |
| Visual boundary context |70.73%| Superseded |
| Candidate boundary ranking |72.56%| Superseded |
| Visual boundary-window HGB |73.48%| Released default0.5.0 |
| Initial long activity pool |74.70%| Historical, before source-join correction |
| Strict source-partition long pool |74.39%| Development-selected research candidate |
| Strict short+long pool |75.91%| Unselected confirmation row; not a declared winner |
| Interaction + long pool |73.78%| Not better than simple pool |
| Fixed original start + long end pool |74.39%| Research ablation |

The initial pool used a 50 ms clip-duration tolerance that could borrow a neighboring
source's first frame. The strict rerun assigns every observation exactly once at
known source starts, requiring ≤20 ms clock agreement. Long-only remains selected;
confirmation drops 74.70→74.39% (244/328 GT; three more than the released baseline).
The earlier unselected short+long 76.52% row is historical, not the corrected result.

The strict pool adds 66 inputs to the original 848-input boundary head. A 133,065-byte
NumPy prototype reproduces all four corrected confirmation predictions within 1e-6s;
feature differences≤1.2e-7. Pooling+both heads take 0.041–0.045s/fullgame on prepared
inputs, four CPU threads / five repetitions; decode/detector/tracking/base features
and base HGB are excluded. No Torch/sklearn/scipy/joblib inference imports.
Production pack/source-boundary integration and complete-pipeline speed remain
unverified. It is not the Studio default.

Fixed median fusion reached72.87% joint and0.488s matched mean boundary error.
Protective fusion reached68.90% joint while reducing lost action0.545→0.281s and
protective cost1.405→1.065s; extra context≈1s. These are timing/safety tradeoffs,
not95% accuracy. Retain both absolute-error and cut-in-cost metrics.

## Compact neural boundary heads — 14 November

21 samples (±4s at 2.5 Hz),31 visual signals plus nested upstream probability;
separate start/end heads, training-only normalization and source-clamped windows.
16 epochs, AdamW 0.003, weight decay 0.01, seed 17, positive weighting ≤12.
All confirmation P/R remains92.05/98.78. No new backbone, decode or audio.

| Model | Development joint | Confirmation joint | CPU seconds/game, prepared inputs |
| --- | ---: | ---: | ---: |
| Frozen HGB |75.06%|73.48%| See current model status |
| MLP32 |67.65%|73.17%|0.021–0.023|
| Bidirectional GRU16 |75.06%|74.70%|0.064–0.069|
| TCN16, dilations1/2/4 |72.10%|73.17%|0.027–0.030|
| Soft-target GRU16 |75.31%|72.87%|0.061–0.077|

Soft-GRU was selected before confirmation, so the unselected hard-GRU 74.70% row
is not a promoted winner. Four CPU threads/five repetitions, source Torch 2.14.1;
all 16 model/game CPU outputs match saved CUDA intervals within 1e-6s. These are
small architecture screens, not exhaustive optimization or full MS-TCN++/ASRF
reproductions. Training used CUDA; all eight final heads occupy about 1.1 MB with
reports. Earlier LSTM16/32 classifiers are in the preceding table.

## Historical detector measurements — 5 November

Eight copied 1920×1080 frames,1280 input / 736×1280 tensor, confidence.25, person class.
CPU FP32, GPU FP16; i7-11700F/RTX3060Ti on Linux. Median three warm passes after
one warmup, including preprocessing/NMS, excluding export/download/startup.
This table preserves all nine detectors from the45configuration comparison.

| Model | CPU torch, batch 1 | CPU torch, batch 8 | CPU ONNX, batch 1 | CPU OpenVINO, batch 1 | GPU torch, batch 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 62.0 | 65.2 | 92.6 | 55.6 | 8.1 |
| YOLOv8s | 148.5 | 162.3 | 179.4 | 134.5 | 11.3 |
| YOLOv8m | 344.0 | 351.6 | 343.2 | 305.6 | 19.1 |
| YOLO11n | 60.3 | 65.9 | 75.2 | 46.9 | 8.1 |
| YOLO11s | 142.0 | 145.0 | 135.4 | 109.4 | 10.8 |
| YOLO11m — historical baseline | 349.2 | 368.0 | 502.5 | 288.3 | 18.1 |
| YOLO26n | 56.0 | 61.0 | 40.3 | 39.0 | 7.3 |
| YOLO26s | 134.4 | 134.1 | 107.0 | 98.1 | 9.7 |
| YOLO26m | 325.9 | 353.9 | 307.1 | 275.1 | 16.4 |

Values are ms/frame. An18.01s detection-only excerpt at 5 Hz measured YOLO11m Torch
batch8 0.54x,11s Torch batch1 1.29x,11s OpenVINO1.49x,26n OpenVINO2.63x.
These are not current whole-pipeline results. At960,11m field-box agreement was
74.4% of its1280 output and11s70.5%; both lost a visible distant player.
Agreement with another detector is not accuracy. Current1280 is a quality baseline,
not proof it is optimal for every venue. Source versions: Python 3.12.3,
Torch 2.14.1+cu130, Ultralytics 8.4.183, ONNX Runtime 1.31.1, OpenVINO 2026.4.1.

## Separate historical confirmation and speed

Before the current boundary refiner, the frozen GPU pipeline scored 95.68%
precision/98.52% recall (133/135 plays) on NT17; the retained Transformer scored
92.81/95.56%. This separate game does not establish 95% precision on the four-game
aggregate. Its source videos are now placeholders, so this is historical evidence.
An older complete PAN run measured 15.86× GPU / 6.59× CPU; current 120 s refiner sample
measurements are 6.43× CPU / 13.95× GPU. Different scopes are not speed regressions or
improvements without a controlled rerun. Full comparison inputs remain indexed.

## Evidence index and reopening a rejected approach

Raw predictions, folds, seeds, input/code/weight hashes, failed attempts and
checkpoints remain in sibling `PlayChop-Experiments`; this cleanup does not delete
them. Paths below are relative to that directory. Never overwrite a prior run.

| Evidence | Location |
| --- | --- |
| Original reviewed snapshots, detector/preprocessing ablations, native CPU/GPU confirmation | `2026-11-06-8705p479/` |
| Person/S3D/MobileNet study, copied frozen features and retained Transformer | `approach-study-81sxifad/` |
| Classifier development and all-family confirmation | `classifier-study-2026-11-10/`, `classifier-confirmation-2026-11-10/` |
| Full TAL adapters, sources/licenses and weights | `action-models-2026-11-10/` |
| Earlier418visual configuration rows and36retained-head parity checks | `joint-boundary-target-expanded-2026-11-10/` |
| Frozen0.5.0 quality/speed and live release verification | `visual-refiner-release-2026-11-13/` |
| Initial activity, interactions/verifiers and end-head ablations | `boundary-temporal-pool-2026-11-13/`, `boundary-interaction-pool-2026-11-13/`, `boundary-end-pool-tuning-2026-11-13/` |
| Corrected pool, numeric weights and parity/speed | `boundary-source-pool-2026-11-13/` |
| Combined228rows, CSV/JSON/plot, including repeats | `visual-feature-pool-confirmed-2026-11-13/` |
| Four new neural configurations, all folds/heads and CPU parity | `boundary-neural-context-2026-11-14/` |
| Compact machine-readable historical pipeline results | [approach results](research/approaches-2026-11-07.json) in this repo |

Configuration rows include repeated baselines, margins and thresholds; they are
not distinct model architectures. Reopen a rejected row only with a stated
change (new representation/data/loss/runtime), a development-only selection
protocol and an expected measurable benefit. A changed threshold on the same
confirmation labels is not fresh evidence.

Useful primary references: [TSM](https://arxiv.org/abs/1811.08383),
[MoViNets](https://arxiv.org/abs/2103.11511), [VideoMAE](https://arxiv.org/abs/2203.12602),
[ActionFormer](https://arxiv.org/abs/2202.07925), [TriDet](https://arxiv.org/abs/2303.07347),
[TCANet](https://arxiv.org/abs/2103.13141), [MS-TCN++](https://arxiv.org/abs/2006.09220).
These motivate experiments; published scores do not establish Scout quality/speed.
