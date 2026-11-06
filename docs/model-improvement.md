# Decision-model improvement experiments

Started 6 November 2026. Priority: represent reviewed play intervals accurately
and make full processing efficient on CPU. Targets supplied by the user are
15× realtime on the local GPU and 5–10× on CPU; these are targets, not promises.
Existing game footage, Scout state, reviews, fields, caches and production models
remain unchanged. All data copies, trained candidates and generated predictions
live in a separate temporary experiment directory, outside the repository.

## Available reviewed data

Initial read-only inventory: seven games have current review MLTs and cached
5 fps detections, totaling 578 play ranges across 31 original videos, about 449
minutes. W3_SAI-HON and W3_SAI-PAN are incomplete with Nextcloud placeholders;
they are excluded until their reviews and complete sources are available.
Authoritative labels are current `Scout/review.mlt` files, not backups or prior
prediction checks. The labels describe play intervals, not individual person
bounding boxes. They support classifier training and end-to-end play evaluation;
they do not directly supervise a person detector.

## Protocol fixed before model comparisons

1. Snapshot reviews, field polygons and detection caches with source/copy hashes.
   Validate original source duration/order against the reviewed timeline. Rebuild
   cache timestamps from that order; never inherit old cache timeline offsets.
   Skip pending downloads and changing/missing files. Record ambiguous caches or
   unverifiable detector provenance; do not silently choose different outputs.
2. Reserve both FLAW_GD4 games (173 ranges) for final evaluation. Development uses
   the other five complete games (405 ranges). Development folds hold out whole
   match days: W2's two games share one group. No adjacent frames or camera files
   from the same game are split randomly between training and validation.
3. First reuse copied detections to compare decision models and 5/2.5/1 fps
   sampling. Recompute tracking/features at each rate instead of thinning already
   computed features. This screening cannot establish the quality of a different
   detector or lowered image resolution; those require fresh detections later.
4. Compare fixed HGB settings before tuning thresholds. Select using development
   folds only. Evaluate preselected quality/CPU candidates on the reserved games
   without retuning on their results. Keep training-fit and held-out scores distinct.
5. Report event precision/recall/F1 at temporal IoU 0.3 and 0.5, missed/extra plays,
   start/end mean absolute error and 90th-percentile boundary error. Legacy
   any-overlap scores are insufficient: one-second overlap can conceal a badly
   located clip. Match intervals one-to-one, maximizing hit count then overlap.
6. Expand to detector size, input resolution, sampling, precision/runtime and
   coarse-to-fine video strategies. Measure complete processing, including decode,
   model load, feature extraction, classification and review creation. Keep cold
   startup/export costs separate from warm repeated processing measurements.

Every run records exact configuration, versions, identities, data split, results
and limitations in its private experiment manifest. Only aggregate findings and
reproducible code enter the repository; no GT, footage or private trained model is
published. Production defaults change only after measured quality/speed tradeoffs
are reviewed. No model is assumed equally accurate because it is faster.

## Work log

- Read-only inventory and review parsing completed: 578 current range labels.
- Added isolated detection subsampling and stricter interval metrics with tests.
- Preparing copied-cache screening; production cache identity cannot be inferred
  from filenames alone, so legacy-cache provenance is explicitly audited.
- Literature review started: efficient temporal video recognition and temporal
  action localization, with primary papers recorded below as they are assessed.

## Literature and implications

- [TSM, ICCV 2019](https://arxiv.org/abs/1811.08383): temporal shifts add temporal
  context to 2D video recognition with little extra arithmetic. It suggests a
  lightweight temporal video baseline, but the backbone still costs compute and
  its recognition results do not establish football play-boundary accuracy.
- [MoViNets, CVPR 2021](https://arxiv.org/abs/2103.11511): efficient video networks
  and streaming feature buffers target video recognition with bounded activation
  memory. A pretrained compact encoder is a candidate alternative to per-frame
  person detection. Our interpretation: it still needs binary play supervision,
  temporal boundary logic and CPU measurements on these camera views.
- [ActionFormer, ECCV 2022](https://arxiv.org/abs/2202.07925): multiscale temporal
  features, local attention and boundary regression address interval localization
  directly. Its benchmark results are not end-to-end Scout speed estimates;
  video feature extraction must be included in any comparison. A small temporal
  head over cheap features is worth testing before adopting a large video model.

These papers motivate experiments, not a model switch. Existing cached detections
provide an inexpensive first check of whether temporal sampling can be reduced
without unacceptable loss. New video encoders, resolution changes and detector
variants will be scored under the same held-out-game protocol and full cost model.

## Primary objective clarified by the user

The primary metric is **event recall with any positive temporal overlap**,
matched one-to-one: aim for 95–98% of GT plays found. Report each game's recall
alongside the aggregate. The secondary metric is timestamp quality on matched
plays: start/end MAE, 90th-percentile error, both boundaries within ±0.5/1/2 seconds,
mean temporal IoU, extra included footage and truncated play time. Precision,
false clips and strict-IoU results remain visible to discourage recall gaming.
Touching endpoints without positive-duration overlap does not count as detection.

This refinement follows the initial strict-IoU screen; its results are retained,
not retrospectively relabeled. Selection now uses development precision among
candidates meeting 95% aggregate event recall, or maximum recall if none qualify.
A below-target result is not an accepted model. Per-game failures remain explicit.

## First screening results

Copied legacy 5 fps detections; five development games, grouped by match day.
The actual detector identity of legacy caches remains unauthenticated. Field
masks use copied polygons and read-only reference/midpoint frames. Duplicate
W3_SAI-DUC caches were required to have identical detection arrays. Timestamps
were rebuilt from the corrected review's source order.

| HGB settings | Sampling | Development F1, IoU≥0.5 |
| --- | ---: | ---: |
| Existing hyperparameters, refitted |5fps|93.45%|
| Compact/regularized |5fps|90.91%|
| Existing hyperparameters, refitted |2.5fps|91.14%|
| Compact/regularized |2.5fps|89.17%|
| Existing hyperparameters, refitted |1fps|55.49%|
| Compact/regularized |1fps|49.16%|

On 173 reserved GD4 ranges, the 5 fps refitted candidate found 171 with positive
overlap: 98.84% recall, 93.44% precision, 12 extra predictions. On those matches,
start MAE was 0.274 s and end MAE 0.614 s; 78.36% had both boundaries within 1 s.
The 1 fps candidate found 152/173: 87.86% recall and 80.0% precision, with worse
boundaries. These measurements use the same reference detections and do not
establish a speedup or accuracy for a smaller detector.

Development is harder: the 5 fps candidate's aggregate event recall was 94.57%,
and the weakest game's recall was 81.67%. The held-out GD4 result therefore does
not establish that the user's recall target is met across all games. Additional
training/calibration must address that failure rather than hiding it in averages.

## Active next experiments

- Add balanced class weighting to focus HGB learning on relatively rare play
  frames; default production training behavior remains unchanged.
- Save out-of-fold probabilities for threshold/hysteresis calibration using
  development games only. Reserved game features need not be read in this mode.
- Generate fresh development detections for YOLO11s at 960 px and YOLO26n at 1280 px,
  at 5 fps, CUDA FP16/batch 16, with exact weight identities and new private caches.
  Refit the decision model for each detector: do not assume reference-detector
  features transfer unchanged. Use subsampling to test lower temporal rates.
- Inspect frame-based tracker gates/confirmation/smoothing as a possible cause of
  low-rate quality loss. No tracker changes have been deployed.
- Read full TSM/MoViNets/ActionFormer papers. TSM's implementation discussion
  highlights memory-movement costs despite low arithmetic overhead; MoViNet's
  streaming buffers address activation memory; ActionFormer's experiments use
  pretrained video features, whose extraction cost must be included here.

All candidates and probability arrays remain private experiment artifacts.
No production classifier, cached detections or edited review has been replaced.

## Broader model families and current evidence

The project is not committed to person detection. The comparison now includes a
**motion-only model** (small-frame optical-flow statistics, no person detector)
and a **direct S3D video model** (pretrained spatiotemporal convolutions with a
new binary/temporal decision head). MoViNet/TSM remain literature-backed next
candidates, not models already validated in this experiment. TorchVision's
[S3D implementation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.s3d.html)
provides a distinct, reproducible non-YOLO baseline using Kinetics400 weights.

Initial synthetic S3D feature timing on the local machine: a 16-frame 112 px clip
took 64 ms on CPU (eight threads, batch 1); 224 px took 248 ms. These numbers exclude
decoding, normalization and the temporal/binary head. They justify testing the
112px alternative, but do not establish full-pipeline realtime performance or
play accuracy. The private development extraction uses shared decoding for both
motion statistics and S3D logits, so its timings are not isolated model benchmarks.

A qualitative audit found an input failure in the weakest game: several GT plays
had people detected in raw boxes but **zero people admitted by the static field
mask**. A camera-homography diagnostic on one such frame retained 11 players where
the existing mask retained 0, using 316 matches and 92.4% RANSAC inliers. This is one
private diagnostic, not a validated camera-correction feature. It explains why
threshold reduction alone cannot recover every play. Saved polygons remain intact.

Threshold/class-weight calibration on development data raised overall overlap
recall toward 98%, but still failed the 95% minimum on the weakest game and often
added many false clips. Those results are not accepted candidates. New visual
features and camera-robust region handling must address missing information.

Split clarification: GD4 remains excluded from training, but its baseline results
have already been inspected and that use is recorded. HON/PAN remain entirely
untouched confirmation games while downloading. Further threshold calibration
reads development out-of-fold probabilities only and rejects reserved files.
Experimental feature schemas cannot be exported as ordinary production packs.

The camera experiment now recomputes tracking features using a union of the
original region and accepted reference projections at five fixed, label-independent
times per source. It preserves the original polygons and compares on development
games only. Fresh YOLO11s 960 px detections have also completed on all five
development games; YOLO26n 1280 px extraction continues. These known-weight
datasets replace ambiguous legacy-cache provenance in subsequent comparisons.
Each comparison now has its own probability directory so independent experiments
cannot accidentally overwrite or mix calibration inputs.

Extraction can now run motion-only without importing Torch, or S3D-only without
computing optical flow. `--decoder cpu` explicitly disables GPU decoding; this
is required for honest CPU-only measurements. These modes avoid charging a
candidate for work belonging to another model. Feature-extraction timings still
exclude model loading and the final temporal head until an end-to-end benchmark
includes those stages.

An additional non-person candidate combines motion and S3D features: motion
provides a 5 Hz clock for boundaries, while the nearest 1 Hz S3D sample supplies
visual context. Fusion requires identical source identities, labels and splits;
it rejects clocks farther apart than one right-hand sampling interval. Only
development arrays are read. It is evaluated independently rather than assumed
to outperform either component.

HON/PAN downloads are now complete. Their labels and footage remain unused for
selection; they are available for additional confirmation after candidate lock.
They share the W3 match day with a development game, so GD4 remains the separate
match-day holdout rather than claiming all confirmation games have unseen venues.

### Development screening update

Whole-day out-of-fold scores at the original default decision threshold:

| Candidate | Event recall | Weakest-game recall | Precision |
| --- | ---: | ---: | ---: |
| Fresh YOLO11s 960, original regions, 5 Hz | 93.83% | 81.67% | 90.69% |
| Fresh YOLO11s 960, camera-region union, 5 Hz | 94.57% | 85.00% | 91.63% |
| Motion statistics, compact head, 5 Hz | 61.48% | 18.99% | 68.22% |
| Broad-region motion, compact head, 5 Hz | 55.80% | 20.25% | 64.57% |
| Motion + frozen S3D112 logits, compact head, 5 Hz | 46.17% | 23.23% | 71.10% |

None meets the target. The costly balanced-head extension of the last row was
stopped after its complete compact-head result; it has no validated result.
All extracted features and complete candidate outputs were retained. These
negative results reject these specific frozen-feature configurations, not video
models in general. The direct S3D-only comparison continues separately.

The next person-feature ablation jointly removes the field mask and lowers
minimum player height from 3% to 1.5% of image height. It reuses existing private
detections rather than decoding/detecting the footage again. If promising, the
two changes must be separated before making a causal claim or choosing defaults.

An initial CUDA-resize compatibility probe produced the expected 15 RGB frames
from three seconds of footage without errors. This establishes compatibility
only; throughput and equivalence to the current decoder remain unverified.
[X3D's official model zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md)
also provides a smaller video-model candidate, including an optimized CPU model.
Its published benchmarks are not Scout play accuracy or end-to-end speed evidence.

### Proposal and preprocessing experiments

On the fixed development folds, camera-union YOLO11s with a compact head,
threshold 0.1, 0.6 s smoothing and a 4 s minimum proposal duration reaches
**395/405 event hits (97.53%), minimum game 95%, precision 78.69%**. Refining
proposal boundaries using their own peak improves boundary p90 to about 1.25 s
and mean IoU to 0.838, while preserving those event counts. This is a development
selection score, not a final confirmation result. The extra 107 clips are still
too many to declare a finished improvement. All tested durations, peak filters
and boundary refinements are recorded in private calibration reports.

Two targeted changes follow from the error analysis:

- A dynamic-region variant interpolates reference-projected field polygons
  between seven fixed keyframes, filters boxes in source coordinates, and falls
  back to the original region if all projections fail. This avoids admitting
  everyone inside the union of every camera position at every time. Saved field
  files remain unchanged.
- A sparse-sampling tracker ablation expresses association gates, smoothing,
  track lifetime and confirmation in elapsed time. The current tracker uses
  some frame-based settings: two confirmations take 0.4 s at 5 Hz but 2 s at
  1 Hz. The ablation exactly preserves the 5 Hz settings and tests 1/2.5 Hz
  features from existing detections. It is not enabled in production.

The pre-scaled YOLO26n 640 experiment resizes on CUDA before RGB transfer, then
maps boxes back to original source geometry before field filtering and tracking.
It keeps its own caches and experimental schema. Some extraction segments exceed
15x realtime, but shared workload, cold-start effects and excluded final-head
cost mean these are not accepted end-to-end benchmarks. GT quality is still under
evaluation; CPU-only acceleration remains to be measured on a selected pipeline.

Validation at this stage: full local check passed 278 tests (four skips) and
source/wheel packaging passed; focused research checks also pass. Those
checks establish code/fixture behavior, not real Windows GPU performance or
the accuracy/speed targets.

### Fixed candidate and full-pipeline confirmation

The next candidate is locked before its reserved-game run: **YOLO26n 1280,
2.5 Hz, dynamic field regions, time-scaled tracking, reference HGB head**.
Its development selection uses threshold 0.2, minimum proposal duration 3 s,
0.2 s smoothing, and half-peak boundary refinement with 0.2 s padding.
Development overlap recall is **395/405 (97.53%)**, weakest game **95%**,
precision **88.37%**, boundary p90 **1.287 s**, and mean IoU **0.825**.

The head is exported once as numeric trees inside a research-specific wrapper.
Both backends use that exact head. The wrapper locks source/GT/split identity,
detector weight hash, sampling, feature order and temporal offsets; ordinary
production classifier loading rejects it. Reserved arrays never enter fitting.
Full-game confirmation joins features in original source order before temporal
prediction, rather than treating each camera file as an independent game.

Three 180 s Linux timing windows measured **7.62x CPU** including detector loading,
CPU-only decoding, camera correction, tracking, features, prediction and
postprocessing. OpenVINO explicitly reported CPU execution and CUDA was disabled.
GPU aggregate timing was **13.40x** including cold initialization; the two later
windows measured **15.29x/15.39x**. Two windows were before the first marked play,
so these are timing evidence, not quality evidence. Full reserved games now test
active play, quality and longer-run throughput together.

The first reserved full game completed on GPU: **84/86 hits (97.67%)**, **90.32%
precision**, boundary p90 **1.570 s**, **17.15x realtime**. The second game and
CPU confirmation remain in progress. Do not report the complete reserved test,
or the application integration, as finished from this partial result.

The 640 px YOLO26n candidate was rejected: default-threshold recall fell to
82.47% at 5 Hz. Calibration recovered high overlap recall only with about 55%
precision. This directly argues against selecting resolution on speed alone.
Frozen S3D112 also failed: direct linear-head recall was 50.86%; its completed
balanced tree head reached 32.84%. These are specific negative configurations,+not a rejection of every video architecture.

New research checks cover frozen-head split/weight guards, source-coordinate
mapping, decoder restoration after errors, and reserved source concatenation
without fitting. Sixteen focused checks pass. The last full suite passed 278
tests with four skips; a final full run follows timed confirmation so it does
not compete with CPU measurements. No production defaults have been changed.
