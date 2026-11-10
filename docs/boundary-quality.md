# Play and boundary quality targets

Updated 10 November 2026. Research targets, not achieved release guarantees:

- Play precision and recall: aim for 95–98%; retain at least 95% recall in
  every measured game. Precision remains the priority.
- Timing: at least 95% of **all GT plays** have **both** start and end within
  ±1 second. A missed play fails this measure. Report each end separately too;
  averaging the two success rates does not establish joint accuracy.
- Earlier starts and later ends are preferable to losing action, but large
  padding is not a substitute for accurate boundaries.

Use the existing one-to-one positive-overlap matching, with IoU≥0.5 results
alongside it to expose oversized clips. False positives count against event
precision. Also report timing precision: accurately bounded matches divided by
all predicted clips. A GT and a prediction can each match only once.

## Continuous measures and directional cost

Let `s = predicted_start - GT_start`, `e = predicted_end - GT_end`, in seconds.
For matched plays:

```
lost  = max(s, 0) + max(-e, 0)
extra = max(-s, 0) + max(e, 0)
absolute boundary error = (abs(s) + abs(e)) / 2
protective boundary cost = (4 * lost + extra) / 2
```

Lower is better. Both costs use seconds; the latter charges a cut into the play
four times as much as equal extra context. This is an initial research weight,
not evidence that four is optimal. Report the components separately. Matched-only
means must accompany recall: dropping difficult plays can otherwise improve them.

For a continuous score that includes misses, sum
`exp(-(abs(s)+abs(e))/(2*tolerance))` across matches and divide by all GT plays.
Perfect matches score one, closer boundaries always score higher, and misses
score zero. A protective counterpart replaces the numerator of the exponent by
`4*lost+extra`. These are auxiliary scores, **not percentages within one second**.
The primary timing acceptance remains the explicit joint ±1-second success rate.

## Reproducible comparison

`scripts/report-classifier-benchmark.py` computes these diagnostics from frozen
predictions. `scripts/compare-boundary-approaches.py --report REPORT [--report
REPORT ...] --manifest MANIFEST --out NEW_DIRECTORY --plot` compares saved
classifier, boundary-regression and fusion reports covering identical games.
It writes JSON, CSV and an optional PNG; matplotlib is research-only. Outputs
record input hashes and never overwrite existing result directories.

Retain original configurations, training folds, seeds, source revisions,
checkpoint hashes, prediction arrays and CPU/GPU timing with every approach.
Choose parameters on whole-day development folds before confirmation. Current
confirmation games have already been inspected repeatedly: they support research
comparison, not a new blind generalization claim. A new untouched game/day is
needed before promoting a winner with confidence.

Boundary-specific context and a separate refinement stage are promising research
directions, rather than merely enlarging the play classifier. Relevant primary
work includes [TCANet](https://arxiv.org/abs/2103.13141), which aggregates local
and global temporal context for proposal refinement, and
[TriDet](https://arxiv.org/abs/2303.07347), which models relative boundary
distributions. Scout experiments inspired by them are not reproductions of
their published benchmarks.
