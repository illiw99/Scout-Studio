# Reproducing classifier comparisons

HGB is the approved classifier, not Random Forest. MLP, AdaBoost, Random Forest,
Extra Trees, logistic models, LSTMs, alternative boosting and full temporal heads
have already been compared. See the [research ledger](research-ledger.md) before
proposing another run; this page no longer treats completed trials as proposals.

`scripts/benchmark-classifiers.py` compares ten bounded variants on the same 31
prepared person columns, nine temporal offsets, five development games and
whole-day folds. Imputation/scaling fit within each fold; MLP uses at most 80 Adam
epochs without random-frame internal early stopping. OOF predictions select
interval settings per family, rather than applying a universal 0.5 threshold.

```bash
python scripts/benchmark-classifiers.py --study /isolated/approach-study \
  --out /new/classifier-study --threads 4
python scripts/report-classifier-benchmark.py --report /new/classifier-study/report.json \
  --manifest /isolated/approach-study/manifest.json --out /new/timing-analysis
```

Inputs: immutable `person/*.npz`, `manifest.json` and `baseline.json`; windowed
feature names must match the frozen baseline. Output must be new and separate.
Reports include code/dependency/input hashes, folds, OOF/confirmation probabilities,
all operating points and a selection written before confirmation reads. Input
hashes are checked again afterward. CPU head timing uses three warm passes/four
threads; it excludes feature preparation/decode/detection and is not throughput.

Use [the evaluation protocol](model-improvement.md) and [metric definitions](boundary-quality.md).
Any non-HGB replacement requires an explicit validated numeric/ONNX runtime;
do not install an unvalidated pickle or change the default from a single score.
