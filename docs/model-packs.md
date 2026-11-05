# Inference model packs

Model packs are separate ZIP files, installed under application user data rather
than beside a game. Importing or activating one does not scan footage, rerun
predictions, or change saved game configuration. The desktop Models menu imports,
activates and rolls back packs; choose **Installed model pack** for new detection
jobs. Running jobs keep their selected version. The default reference detector
and experimental faster CPU profile remain available.

No production baseline pack is included yet. A baseline needs reviewed model
hashes, redistribution permission, source/license attribution and held-out-game
quality approval. Synthetic test graphs are never distributed as production
models. A smaller detector's speed does not establish equivalent play accuracy.

Packs contain exactly `manifest.json`, `classifier.json`, `LICENSE.txt`, and either
`detector.onnx` or `detector.xml` with `detector.bin`. Pickles, checkpoints, scripts,
training provenance and arbitrary model-directory contents are excluded. The
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
