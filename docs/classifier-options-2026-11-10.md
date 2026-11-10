# Classifier options — 10 November 2026

The approved decision head is HistGradientBoosting (HGB), not Random Forest.
The pipeline extracts person/motion/formation features and their temporal context,
then converts head probabilities into play intervals. These classifier candidates
reuse that representation; they do not replace the person detector.

## Measured evidence versus proposed comparisons

The [previous comparison](play-approach-comparison-2026-11-07.md) tested a person
feature transformer and temporal CNN variants, plus frozen S3D/MobileNet visual
encoders. The accepted HGB pipeline beat the selected transformer on NT17-SAI:
95.68% precision / 98.52% recall versus 92.81% / 95.56%. This does not establish
that a simple MLP or another tree ensemble would lose. No completed comparable
AdaBoost, MLP, Random Forest or Extra Trees study was found in the recorded
results. No new fitting or held-out evaluation was performed for this research.

## Shortlist

| Candidate | Reason to compare (hypothesis) | Initial bounded search |
| --- | --- | --- |
| HGB | Reproduce the current baseline; regularization may reduce false clips | Depth 2/3/4, 100/200 iterations, minimum leaf 20/60, L2 0/1 |
| Small MLP | Nonlinear feature interactions with a compact dense CPU head | 32, 64, or 64→32 hidden units; ReLU/tanh; L2 0.001/0.1; fixed epoch budget |
| Extra Trees | Different randomized partitions may generalize across camera conditions | 64/128 trees, depth 6/12, minimum leaf 5/20 |
| Random Forest | Establish an actual forest baseline | Same tree/leaf budgets as Extra Trees |
| AdaBoost | Test whether shallow boosted rules suffice | Tree depth 1/2, 50/150 estimators, learning rate 0.1/0.5 |
| Logistic regression | Cheap linear baseline to quantify the need for nonlinear complexity | Standardized features, L2 regularization |
| CatBoost / LightGBM / XGBoost | Alternative regularized boosting implementations | Shallow, bounded models after the sklearn comparison |

These are research proposals, not selected model settings or performance claims.
Start with a small coarse search and extend only promising families. Extra
packages belong in a separate experiment environment, not the shipping app.

Scikit-learn recommends scaling MLP inputs and fitting that scaling on training
samples only. Its MLP has no GPU implementation; a small CPU head is still a
reasonable candidate for this application. Adam and L-BFGS are worth comparing
within a runtime budget; correlated frame count does not equal independent game
count. [MLP guide](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

AdaBoost successively reweights samples, increasing focus on misclassified ones.
With imperfect GT, inspect whether this magnifies annotation errors; that is a
risk to test, not an established result on these games.
[AdaBoost API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
Extra Trees randomizes tree splits; its benefit here remains unmeasured.
[Extra Trees API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

Alternative boosting libraries also require complexity control. CatBoost documents
Ordered boosting as potentially better on small datasets with a speed tradeoff;
LightGBM warns about unconstrained leaf-wise growth; XGBoost documents depth,
regularization and sampling controls. None promises a Scout improvement.
[CatBoost](https://catboost.ai/docs/en/references/training-parameters/common),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html).

## Fair experiment and promotion gate

1. Reuse hash-verified feature arrays and exact GT snapshots read-only. All new
   arrays, reports and fitted objects live in a fresh experiment directory.
2. Hold out whole games and preferably entire match days. Fit scalers, feature
   selection and classifiers on training folds only. Do not use random frame
   splits or random internal early stopping across adjacent frames.
3. Give each family the same temporal inputs, fold membership and comparable
   tuning budget. Use development out-of-fold probabilities for interval tuning;
   probability scales differ, so one common 0.5 threshold is not a fair test.
4. Maximize one-to-one event precision subject to at least 95% recall in every
   development game. Also report false clips, missed plays, IoU≥0.5 recall,
   boundary p90, model size and CPU prediction time with a fixed thread budget.
5. Freeze the choice and parameters before held-out scoring. Existing validation
   games and NT17-SAI have already been inspected; they are repeated confirmation,
   not a fresh blind test for a newly selected classifier. Seek a new untouched
   game/day for stronger generalization evidence.
6. Measure complete CPU/GPU pipeline time before claiming a speed improvement.
   A faster head cannot remove detector/decode work. A head swap alone has no
   established path to a large total-speed gain without profiling.
7. Promote only after quality, export parity and standalone engine compatibility
   pass. The existing numeric JSON exporter is HGB-specific; other families need
   a verified numeric or ONNX export and explicit runtime support. Do not replace
   the approved default with an unvalidated pickle.

Further hypotheses: inspect false-positive clips to design negative examples;
compare a compact temporal CNN with the same features; test a small sequence
smoother with separately reported boundary effects. Ensembling is a later option
only if out-of-fold errors are complementary. These are not measured improvements.

## Reproducible benchmark

`scripts/benchmark-classifiers.py` implements ten bounded variants across six
families. It consumes the previous approach study's prepared `person/*.npz`,
`manifest.json` and frozen `baseline.json`. Every family gets the same 31 feature
columns, nine offsets, five development games and whole-day folds. Imputation
and MLP/logistic scaling fit inside each training fold. MLP uses 80 Adam epochs
at most, without random internal early stopping; warnings are retained.

```bash
python scripts/benchmark-classifiers.py --study /path/to/isolated-approach-study \
  --out /path/to/new-classifier-study --threads 4
```

The output directory must be new and outside the input study. Four excluded
columns and feature expansion are already represented in the prepared inputs;
the script validates their exact windowed names against the frozen baseline.
It saves file/code/dependency hashes, fold membership, OOF probabilities, all
development operating points, a selection file written before loading held-out
arrays, and per-family confirmation results. All families' operating points are
frozen before any confirmation scoring. Input hashes are checked again at the end.

Timestamp reporting includes start/end MAE and p90, IoU≥0.5 recall, both boundaries
within 0.5/1/2 seconds, and excess/truncated seconds. `boundary_recall` counts
missed GT plays as failures, while `both_boundaries_within_seconds` describes
matched plays only. Export pre/post padding is excluded. Precision is primary
under the per-game recall floor; boundary recall and IoU break precision ties.
No explicit numerical boundary acceptance threshold has yet been specified.

The user subsequently chose **1 second**, preferring an early start/late end to
losing play action. `scripts/report-classifier-benchmark.py` adds a separate
post-selection analysis: late starts, early endings, early starts and late
endings each get counts/mean/p90; both strict 1-second boundary recall and
coverage within that tolerance use all GT plays as the denominator. Coverage
alone permits overlong clips, so the same table retains IoU and extra seconds.
The frozen development choice is not changed using these confirmation results.

```bash
python scripts/report-classifier-benchmark.py --report /new-study/report.json \
  --manifest /isolated-approach-study/manifest.json --out /new-study/timing-analysis
```

CPU timing is three warm prediction passes with four threads per confirmation
game; the report keeps every observation. This excludes feature preparation,
decode, detector, startup and interval postprocessing. It is not end-to-end
realtime throughput. No trained model is installed or production weight replaced.
