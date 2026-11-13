# Visual feature-pool experiments — 13 November 2026

The strongest development-selected addition in this round increased joint
one-second timing accuracy from **73.48% to 74.39%** after a strict source-join audit, with event precision
**92.05%** and recall **98.78%** unchanged. This is three additional correctly
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
| Strict source-partition long pool | **74.39%** | 95.43% | 77.13% | **0.491s** |
| Initial long pool, before source audit | 74.70% | 95.73% | 77.13% | 0.491s |
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

There are twenty-two learned boundary settings including repeated baselines,
six verifier classifiers with 126 threshold settings, and repeated fixed-margin
comparisons. These are not hundreds of independent model architectures.
The corrected combined report contains 228 configuration rows.

The initial unselected short+long pool row reached 76.52% on confirmation;
its strict source-partition rerun reached 75.91%. Its
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
- `boundary-source-pool-2026-11-13/results/{selection,report}.json`: strict source-partition rerun.
- `boundary-source-pool-2026-11-13/numeric-prototype/`: numeric heads and parity/speed checks.
- `visual-feature-pool-confirmed-2026-11-13/`: all 228 CSV/JSON rows and comparison plot.
- `visual-feature-pool-comparison-2026-11-13/`: the earlier 204-row snapshot, retained.

The initial research head took about 1.73 seconds across four confirmation games
in an unoptimized Python candidate loop. A separate NumPy-only prototype now
reproduces the corrected strict-source candidate's predictions on all four
games. Its 133,065-byte numeric refiner takes median **0.041–0.045s per full
game** over five repetitions (four CPU threads), including pool construction,
candidate features and both heads on prepared inputs. Export/load validation
uses no Torch, sklearn, scipy or joblib in the inference process. Pool values
match the training implementation within 1.2e-7; final intervals match within
1e-6 seconds. This is not an app model-pack format or full-pipeline speed claim.
Decode, detector, tracking, base feature/window preparation and base HGB are
excluded. See [the frozen model's actual CPU/GPU benchmark](visual-refiner-2026-11-13.md).

## Source-join correction

The first pool implementation allowed 50ms beyond a recorded clip duration.
Near a file join that could include the neighboring source's first sample.
The follow-up partitions frames exactly once at known source starts, requiring
clock agreement within 20ms, and repeats all four depth-two pool configurations
with the same nested development selection. The long pool remains selected;
confirmation changes 74.70→74.39%. Original reports and the failed prototype
coverage check remain saved. Production integration must carry exact source
boundaries rather than infer arbitrary joins from an evenly sampled clock.

The next promotion requires production model-pack/feature-contract integration,
full-pipeline speed measurement and fresh independent game/day confirmation.
NT17's seven MP4 entries were checked again on 13 November and remain one-byte
`.nextcloud` placeholders; no automatic hydration was requested or performed.
These incremental pools improve timing modestly; they have not resolved
precision or established the 95% timing target.
