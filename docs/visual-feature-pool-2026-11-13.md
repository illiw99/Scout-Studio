# Visual feature-pool experiments — 13 November 2026

The strongest development-selected addition in this round increased joint
one-second timing accuracy from **73.48% to 74.70%**, with event precision
**92.05%** and recall **98.78%** unchanged. This is four additional correctly
bounded plays among 328 confirmation plays. It remains a research candidate;
Studio 1.0.3 ships the separately frozen 73.48% model. No candidate meets the
95% precision / 95% joint timing targets. No audio was used.

## Comparable selected results

Five development games (405 GT plays), whole-match-day outer folds and nested
upstream classifier fitting. Each study selects settings on development
predictions before confirmation. Four confirmation games (328 plays) have been
inspected repeatedly; this is adaptive validation, not a new blind test. The W3
day occurs in both partitions. Legacy cache provenance limitations remain.

| Approach | Both cuts ±1s / all GT | Start ±1s | End ±1s | Mean absolute boundary error |
| --- | ---: | ---: | ---: | ---: |
| Frozen visual window refiner | 73.48% | 95.43% | 76.22% | 0.505s |
| Long activity pool, both heads | **74.70%** | 95.73% | 77.13% | **0.491s** |
| Interaction + long pool | 73.78% | 94.82% | 76.83% | 0.501s |
| Long pool, fixed original start head | 74.39% | 95.43% | 77.13% | 0.494s |

All rows retain 324 matched plays, four misses and 28 false positives: precision
92.05%, recall 98.78%, minimum game recall 96.15%. Misses fail timing accuracy.
The continuous protective cost charges late starts and early ends four times as
much as extra context; it remains reported alongside symmetric timing accuracy.

## What was tested

- Short (1.2s) and long (3.2s) source-local pools over eleven existing activity,
  speed, coherence and formation features. Each contributes past/future means,
  current-minus-past/future changes, local variation and local peaks. Long pools
  add 66 inputs to the original 848-input boundary head. No additional vision
  backbone, model weights or video decode are required.
- Eleven player-interaction cues: nearest-player spacing, close pairs, relative
  closing motion, connected player groups, stopped/recently stopped players and
  deceleration. Five offsets contribute 55 boundary inputs. Extraction reused
  private native detection snapshots and recomputed field projection; original
  footage, fields, GT, caches and models were preserved. Source-local alignment
  explicitly permits at most one 2.5Hz sample at differing cache endpoints.
- Eight long-pool end-head settings, retaining the original start head: depth,
  target tolerance, regularization, leaf size and boosting iterations. The
  development-selected setting was the original small head with long features.
- Six proposal verifiers: depth-one/two HGB, depth-four/eight random forests,
  and two regularized logistic models. Twenty-one thresholds each were scored
  using nested day-held-out upstream predictions. The selected threshold retained
  every proposal; no precision gain satisfied the development recall floor.

There are eighteen learned boundary settings including repeated baselines,
six verifier classifiers with 126 threshold settings, and repeated fixed-margin
comparisons. These are not hundreds of independent model architectures.
The combined report contains 204 configuration rows.

The unselected short+long pool row reached 76.52% on confirmation. Its
corresponding development score was below the selected long-only row, so that
confirmation result is recorded but is not used to declare a new winner.
The five temporal-pool candidates introduced no new overlap/order conflicts on
any development or confirmation game. All source scripts, failed attempts,
weights, split identities, alignment audits and predictions remain saved.

## Results and speed scope

Detailed local results live in the sibling `PlayChop-Experiments` directory:

- `boundary-temporal-pool-2026-11-13/results/{selection,report}.json`
- `boundary-interaction-pool-2026-11-13/results/{selection,report}.json`
- `boundary-interaction-pool-2026-11-13/proposal-verifier/{selection,report}.json`
- `boundary-end-pool-tuning-2026-11-13/results/{selection,report}.json`
- `visual-feature-pool-comparison-2026-11-13/`: full CSV/JSON and comparison plot.

The long-pool research head took about 1.73 seconds across all four confirmation
games in an unoptimized Python candidate loop. This is not a complete-pipeline
throughput measurement and cannot be compared directly with the deployed
vectorized numeric runtime. See [the frozen model's actual CPU/GPU benchmark](visual-refiner-2026-11-13.md).

The next promotion requires a portable numeric feature/head implementation,
parity checks, full-pipeline speed measurement and fresh independent game/day
confirmation. These incremental pools improve timing modestly; they have not
resolved precision or established the 95% timing target.
