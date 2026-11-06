# Inference model packs

Model packs are separate ZIP files, installed under application user data rather
than beside a game. Importing or activating one does not scan footage, rerun
predictions, or change saved game configuration. The desktop Models menu imports,
activates and rolls back packs; choose **Installed model pack · automatic** for new detection
jobs. Running jobs keep their selected version. The default reference detector
and experimental faster CPU profile remain available.

An experimental inference-only installer can use these packs without Torch or
training libraries; see [runtime validation](runtime-validation-2026-11-05.md).
No production baseline pack is included yet. A baseline needs reviewed model
hashes, redistribution permission, source/license attribution and held-out-game
quality approval. Synthetic test graphs are never distributed as production
models. A smaller detector's speed does not establish equivalent play accuracy.

Packs contain exactly `manifest.json`, `classifier.json`, `LICENSE.txt`, and either
`detector.onnx` or `detector.xml` with `detector.bin`. Pickles, checkpoints, scripts,
training datasets, footage and arbitrary model-directory contents are excluded. The
version1 manifest records ID/version, exact file sizes/SHA-256 hashes, engine,
runtime API, preprocessing, image size, sampling, classifier format, license,
source and postprocessing settings. Runtime API1 supports FP32 CPU COCO80/person0
outputs and numeric binary HGB classifiers, with Scout's exact feature/window
order. The selected engine must be installed; a probe must pass before import or
activation succeeds. Internal hashes detect damage; use a separately trusted
archive SHA-256 to establish which publisher-provided file you intended to import.
Publisher signatures and a hosted model channel are not configured.

Build only from deliberately approved inputs. A reviewed approval JSON is the
complete version1 manifest, including exact input hashes. `settings` contains
`conf`, `model_threshold`, `model_low_ratio`, `model_smooth`,
`model_min_duration`, `model_pre_roll`, `model_post_roll`, `merge_gap`.
`runtime_api` is1, `preprocessing` is `scout-coco-person-v1`, and `classifier`
is `scout-hgb-v1`. Do not automatically approve whatever is in `models/`.

```bash
scout model-pack build --approval approved.json --inputs approved-inputs --out new-pack.zip
scout model-pack import --store /absolute/pack-store --file new-pack.zip --sha256 EXPECTED_HASH
scout model-pack list --store /absolute/pack-store
scout model-pack activate --store /absolute/pack-store --pack baseline@1.0.0
scout --model-packs-dir /absolute/pack-store --model-pack active --match Game run
scout model-pack rollback --store /absolute/pack-store
```

CLI imports never read game configuration. Explicit `--model-packs-dir` supports
portable stores. The native app uses Qt's application-data path; pass that path
explicitly when sharing packs with a CLI on another platform. Installed versions
are immutable through this API: a different pack cannot replace the same version.
Activation atomically records current and previous selections; repeated activation
is a no-op and a failed activation preserves rollback. Old versions stay installed.
Detection caches record pack identity and remain separate from reference caches.

## Version-two adaptive-field profiles

Schema2/runtime API2 packs use `scout-coco-person-resized-v1` preprocessing,
`scout-hgb-v2` classifiers and the `dynamic-field-v1` feature pipeline. They bind
the numeric feature settings to the classifier, restore resized boxes to source
coordinates, project field regions from observed camera frames, and scale tracker
timing with sampling rate. Their settings add `model_peak_threshold`,
`model_refine_fraction` and `model_refine_padding`. Saved polygons are never
replaced. Legacy packs, classifier files and default cache identities remain valid.
Older builds reject version-two classifiers/packs rather than silently using
legacy feature meanings.

These packs keep the same non-executable file allowlist. An optional
`gpu_reference` specifies only an approved built-in detector name and SHA-256.
It cannot choose an arbitrary URL or carry a checkpoint. On a compatible reference
build, automatic mode verifies an official pinned asset before Torch loads a
private snapshot. CPU-only builds do not probe Torch. Missing/incompatible GPU
libraries, asset preparation failure or automatic reference initialization failure
select the pack's CPU graph. Explicit GPU requests report failures. A standalone
segment job can reuse a complete matching CPU cache set after a prior automatic
fallback; it never combines partial CPU/GPU cache sets.

CPU import/probing does not download or initialize the GPU reference. Runtime
selection and loading happen in the processing worker. `cpu_threads` and
`gpu_batch` provide bounded defaults; per-game polygons remain authoritative.
Additional camera samples are stored with new detections, so normal processing
does not need a second source decode for camera alignment. Cached camera clocks
rebase with source ordering just like the raw detections.

`scripts/build-reviewed-pack.py` builds a private pack from an explicit frozen
candidate and a managed graph with matching checkpoint/resolution provenance.
It does not train, activate, scan source libraries or publish anything. The current
reviewed pack is a local evaluation artifact, not a public model release.

Version-three numeric classifiers (`scout-hgb-v3`) may be carried by schema-two
packs. They explicitly bind a bounded, ordered list of temporal offsets and a
known feature recipe: `base-v1` or `play-context-v1`. The latter derives nine cheap
ratios and formation/activity changes from the same raw player observations,
without another detector pass. Import validates the complete expanded column
order; activation restores the recipe and offsets. Older readers reject this
classifier format. Legacy classifiers retain their original short context.

Custom-context PKLs use `scout-context-pickle-v1` and a distinct classifier key so
older readers fail instead of silently applying their fixed time offsets. The
numeric JSON remains the portable inference artifact; matching settings are
required for either format. Existing PKLs are never replaced by these experiments.
