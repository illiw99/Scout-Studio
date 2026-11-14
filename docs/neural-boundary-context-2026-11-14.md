# Compact neural boundary-head comparison — 14 November 2026

A recurrent model is computationally feasible here, but this screen did not
produce a reliable accuracy improvement. The development-selected soft-target
GRU reached **72.87%** joint one-second timing accuracy on confirmation, below
the frozen HGB's **73.48%**. It is not promoted. All weights and failed approaches
are retained; the existing Studio default remains unchanged.

## Fixed experiment

Separate start/end candidate scorers receive 21 samples (±4s at 2.5Hz), each
with 31 existing visual signals plus an upstream play probability. New sequence
windows clamp to source-file boundaries. Training uses five development games
with whole-day outer folds and nested upstream fitting, and training-only
normalization. Four confirmation games are read after development selection;
they have been inspected repeatedly and are not a fresh blind test.

Four configurations were fixed before confirmation: a flattened MLP with 32
hidden units, bidirectional GRU with 16 units per direction, residual temporal
convolution with 16 channels and dilations 1/2/4, and the same GRU trained with
Gaussian soft boundary targets. Training uses 16 epochs, AdamW at 0.003,
weight decay 0.01, seed 17 and bounded positive weighting. There is no
confirmation-based early stopping or threshold tuning. These are compact
research architectures, not exhaustive neural-network optimization.

| Candidate | Development joint ±1s | Confirmation joint ±1s | Selected on development |
| --- | ---: | ---: | --- |
| Frozen HGB boundary head | 75.06% | 73.48% | No |
| MLP32 | 67.65% | 73.17% | No |
| Bidirectional GRU16 | 75.06% | 74.70% | No |
| Temporal convolution16 | 72.10% | 73.17% | No |
| GRU16 with soft targets | 75.31% | 72.87% | **Yes** |

Selection retains the existing per-game recall/precision guardrails, then
maximizes joint timing, continuous timing quality and protective cost. The
hard-target GRU's higher confirmation row is retained but not cherry-picked as
a selected winner. All confirmation rows have event precision 92.05%, recall
98.78% and minimum game recall 96.15%. These heads refine the existing play
proposals; they do not independently solve proposal false positives.

## CPU cost

Four CPU threads, five repetitions per complete confirmation game, on prepared
visual features and probabilities. Times include sequence construction and both
boundary heads, excluding video decode, detection, tracking, base features and
base classifier. CPU predictions reproduce all saved CUDA confirmation
intervals within 1e-6 seconds.

| Architecture | Prepared-feature CPU time per game |
| --- | ---: |
| MLP32 | 0.021–0.023s |
| GRU16 | 0.064–0.069s |
| Temporal convolution16 | 0.027–0.030s |
| GRU16 with soft targets | 0.061–0.077s |

These are source-build Torch 2.14.1 measurements, not pinned-installer or
full-pipeline throughput guarantees. Inference uses CPU explicitly; training
used CUDA. The experiment occupies about 1.1MB including all eight final heads
and detailed fold results. No audio, new backbone, footage copies or generic
weight downloads were used. No original Scout files were changed.

## Evidence and next decisions

Local evidence is in sibling
`PlayChop-Experiments/boundary-neural-context-2026-11-14/`: `PROTOCOL.md`,
`driver.py`, `results/selection.json`, `results/report.json`, all final weights,
`benchmark_cpu.py` and `results/cpu-latency.json`. Reports retain predictions,
fold membership, training loss, source identities and checkpoint hashes.

The architecture screen follows the temporal-convolution motivation in
[MS-TCN++](https://arxiv.org/abs/2006.09220) and the separate boundary-estimation
motivation in [ASRF](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html).
This is an engineering interpretation of those ideas, not a reproduction or a
claim that their published benchmarks transfer to Scout.

The [strict-source pooled candidate](visual-feature-pool-2026-11-13.md) remains
at 74.39% on repeated confirmation. The 95% precision and joint timing aims
remain unmet. A fresh independent game/day is needed before asserting that
incremental improvements generalize; NT17 videos remain Nextcloud placeholders.
