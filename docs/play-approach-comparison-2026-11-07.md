# Play pipeline comparison — 7 November 2026

The accepted `scout-reviewed-precision@0.3.0` remains the default. The new temporal
transformer passes the recall floor but loses precision. Frozen person-free S3D
and MobileNet encoders run quickly but fail the quality requirement. These results
do not establish that a larger custom vision transformer would improve this task.

## Data and selection

Nine reviewed games contain 733 annotated plays. Five development games (405
plays): GD1 TIG–THU, W2 CON/VET, W3 DUC and GD3 INV. Four validation games (328
plays): GD4 CON/RAI, W3 HON/PAN. Development cross-validation holds out whole match
days, grouping the two W2 games together. Learned normalization and head fitting
use only the training folds. Development out-of-fold probabilities select interval
parameters and model variants; final fitting uses the five development games.

Selection maximizes event precision subject to **at least 95% recall in every
development game**. If no candidate qualifies, the maximum minimum-game-recall
candidate is retained as a failed experimental comparison, not promoted. Recipe
building selects across runs using development results only. The selection file
is written before validation arrays are loaded. These validation games have been
checked repeatedly in this project: this is repeated held-out-game validation,
**not a fresh blind test**. HON/PAN share a day with development DUC. Human GT is
useful but imperfect and covers few venues/camera conditions.

Person features reuse the approved study’s cached 2.5 Hz expanded play-context
features, excluding its four excluded feature columns; the same frozen training
arrays were hash-checked. This is not a newly authenticated raw-video extraction
of all person training inputs. Visual features were freshly extracted from the
41 original source clips. Both studies use the same GT/split, not the same feature
representation. GPU-derived validation features are scored by CPU ONNX heads.
Full CPU-derived validation quality for the new recipes remains unmeasured.

## Approaches and tuning

- Person temporal model: 31 named motion/formation/context features, nine temporal
  tokens spanning ±10 seconds; a small two-block attention model trained from
  scratch. This is a feature transformer, not a raw-patch vision transformer.
- S3D: frozen pretrained Kinetics video network, 16 RGB frames at 5 Hz, 112×112,
  400 output logits; a trained temporal head over nine tokens spanning ±4 seconds.
- MobileNet V3 Small: frozen ImageNet frame encoder, 96×96, 576 pooled features,
  2.5 samples/s; a trained temporal head spanning ±4 seconds.

Each approach tried transformer widths 32/64 and a width-32 dilated temporal CNN.
Initial training used 8 epochs (32-wide) or 12 epochs (64-wide), balanced per-game
positive/negative batches, AdamW, seed 17 and train-fold-only learned normalization.
The person transformer also tried 28 and 64 epochs. Visual tuning tried raw versus
unlabeled per-game z-score features, and lower encoder rates (S3D 0.5 Hz, MobileNet
1.25 Hz). Interval searches varied threshold, hysteresis, smoothing, duration, merge
gap, peak filtering and boundary refinement. These are finite searches, not proof
of a global optimum. Backbone fine-tuning and raw-patch ViT training were not done.

Per-game z-score uses only the input game’s unlabeled features, also at inference.
It is an offline full-game transform; it does not fit to validation labels. Visual
encoder outputs are quantized to float16 to reproduce the saved extraction inputs.
The exported heads include training means/scales and use CPU ONNX. Encoder CPU
ONNX exports matched Torch CPU on random uint8 input within 3.34e-6 maximum error.
This is numerical export validation, not a CPU detection-quality guarantee.

## Quality

Event matching is one-to-one with any positive overlap. A huge predicted clip
cannot count as detecting many plays. IoU ≥0.5 measures useful timestamp overlap;
boundary p90 measures error across the boundaries of matched plays.

| Selected approach | Precision | Recall | Worst-game recall | Hits / 328 | False plays | IoU≥0.5 recall | Boundary p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted default, previous GPU-feature validation | 92.05% | 98.78% | 96.15% | 324 | 28 | See prior study | 1.400 s |
| person | 87.53% | 98.48% | 96.15% | 323 | 46 | 92.07% | 2.203 s |
| s3d | 40.24% | 91.16% | 85.90% | 299 | 444 | 46.04% | 6.407 s |
| mobilenet-v3-small | 35.61% | 58.84% | 47.67% | 193 | 349 | 9.45% | 29.699 s |

The default row is the prior accepted result, not a retrained baseline in this
round. Its full CPU-feature validation was 92.53% precision / 98.17% recall. The
new person transformer generated more false clips and worse timestamp boundaries.
The visual variants show that speed alone cannot justify replacing the detector.

| Approach | Development-selected recipe | Development precision | Worst-game recall |
| --- | --- | ---: | ---: |
| person | person-raw-transformer64-2p5hz | 91.16% | 95.29% |
| s3d | s3d-game-zscore-tcn32-1p0hz | 43.20% | 62.63% |
| mobilenet-v3-small | mobilenet-v3-small-game-zscore-tcn32-2p5hz | 34.47% | 44.30% |

The lower-rate visual runs did not improve development minimum-game recall and
were not selected. Longer person training helped through 64 epochs. Further
interval tuning did not beat that run’s development precision under the recall
constraint. Validation outcomes were not used to select the installed recipes.

## Complete runtime

Sequential fresh-process tests used the same 180-second window (480–660 seconds
of the first GD1 source), four inference threads and new private caches. Hardware:
Intel i7-11700F, NVIDIA RTX 3060 Ti, local Linux. No training or simultaneous model
timing ran during these tests. File cache state was not flushed. Each includes
configuration/model loading, real decode, prediction, features/postprocessing and
review writing; MP4 encoding and one-time recipe installation are excluded.

| Pipeline | Runtime | Pipeline seconds | Pipeline realtime | Process wall seconds | Process realtime |
| --- | --- | ---: | ---: | ---: | ---: |
| approved | cpu | 31.41 | 5.73× | 31.66 | 5.69× |
| approved | cuda:0 | 14.51 | 12.41× | 15.34 | 11.74× |
| person | cpu | 32.20 | 5.59× | 32.45 | 5.55× |
| person | cuda:0 | 14.34 | 12.55× | 15.18 | 11.86× |
| s3d | cpu | 16.87 | 10.67× | 17.04 | 10.56× |
| s3d | cuda:0 | 13.37 | 13.46× | 14.10 | 12.76× |
| mobilenet-v3-small | cpu | 11.21 | 16.05× | 11.37 | 15.83× |
| mobilenet-v3-small | cuda:0 | 13.53 | 13.31× | 14.28 | 12.61× |

Pipeline timings exclude interpreter imports; process wall time includes them.
The short window pays startup overhead more heavily than full games. These eight
single-window runs establish functionality and indicative runtime, not a hardware
latency distribution or full-game accuracy. CUDA MobileNet is slower here than
CPU because its tiny encoder saves too little work to repay GPU startup/decode
overhead. Visual decode takes about 10 seconds of this 180-second sample, so the
encoder itself is not the only bottleneck.

The accepted default’s earlier **full 64.27-minute PAN native-worker** checks were
6.59× CPU and 15.86× GPU, meeting the requested local speed range with 96.15% PAN
recall. Keep that scope distinct from these short-window numbers. See the
[prior acceptance evidence](model-improvement.md#completed-local-acceptance).

## Evidence, source safety and next steps

Private experiment: `PlayChop-Experiments/approach-study-81sxifad`. It retains the
frozen manifest, setup/weight hashes, original extractor snapshot, feature arrays,
fold probabilities, calibration/selection files, ONNX exports, reports, runtime
logs and original-integrity receipt. The original 41 video size/mtime identities
and 109 snapshotted review/field/cache hashes still match after the experiments.
No original Scout state was modified. No training data or weights enter Git.

The initial visual extractor materialized one source at a time. Its exact source
snapshot/hash is retained. The shipping runtime and future extraction script use
bounded streaming; deterministic tests compare centered windows and repeated
edges with the original extraction convention. Initial lower-rate attempts failed
on duplicate rows from rounded overlapping source durations; they produced no
accepted models. Source-clock subsampling was fixed and those runs restarted in
new directories. Failed attempts are excluded from timing.

Unused generated build trees/download caches were removed under the user’s
storage authorization: 8.32 GiB reclaimed, with active installs, reusable build
inputs, data and model evidence preserved. No duplicate MP4 collection was made.

All three recipes are experimental. More diverse reviewed games and a new
uninspected test day would improve confidence more than repeatedly tuning these
four validation games. If pursuing person-free models further, use field-focused
higher-detail crops and domain fine-tuning before scaling transformer size. That
is an informed next hypothesis, not an observed improvement.

## Literature

The architecture choices are grounded in [S3D’s factorized spatiotemporal
convolutions](https://arxiv.org/abs/1712.04851) and [MobileNetV3’s hardware-aware
architecture design](https://arxiv.org/abs/1905.02244). [TimeSformer](https://arxiv.org/abs/2102.05095)
shows that space/time attention is a viable video-classification architecture.
Those papers evaluate different tasks/data; their results do not establish Scout
precision, clip-boundary quality or CPU throughput. This study measured two frozen
visual encoders and small trained temporal heads, not TimeSformer itself.

The [implementation and usage guide](play-pipelines.md) describes the complete
pipeline contract, model import, Studio selection and repeatable benchmark CLI.

## Implementation and local deployment validation

The full private CPU/OpenVINO environment check passed **365 tests**, with two
native-platform skips. Source/wheel packaging checks and wheel CLI smoke passed.
An independent pipeline review checked source safety, immutable queued selection,
clock handling, bounded video memory and CPU fallback. Its benchmark relative-path
finding was fixed with a differing-CWD/config-root regression test. Automatic GPU
failure retries the complete game on CPU; all interval parameters are recipe-bound.

The installed Linux reference standalone was rebuilt from `76d5225`. Native X11
smoke passed: GUI, real synthetic playback, export, CLI, worker, OpenVINO, verified
bundled default and runtime dependencies. Every new recipe completed synthetic
CPU and CUDA processing in the **actual frozen worker** (six checks). A native
chooser check shows all three recipes, with the accepted default selected.
The normal project launcher and Applications entry now point to that build.
Restart Studio to load it; existing open sessions are not terminated.

Only inference assets were imported into the user's pipeline store. The active
pack remains `scout-reviewed-precision@0.3.0` and the saved detection profile remains
`active-pack`. The new executable shares immutable dependency/media/model payload
with the previous installation; no dependency version changed. The previous
launcher/desktop entry and deployment receipt remain in the private build evidence.
Unused intermediate build archives were removed after validation (another 183 MiB).
No push, public model upload or release publication was performed.

Offscreen tests and Linux execution do not verify Windows installers, Windows GPU
hardware, macOS execution or full CPU-derived visual-model quality on the games.
