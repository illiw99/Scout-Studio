# Default Scout Studio detection profile

The selected local profile is `scout-reviewed-precision@0.5.0`: YOLO26n at 1280,
2.5 samples/s, adaptive fields, time-scaled tracking, derived play features and
wide temporal context. The profile binds its classifier, detector, sampler and
postprocessing together; replacing only an old PKL is insufficient. Its operating
point prioritizes precision while keeping recall above 95% in every measured
validation game. See [measurements and limitations](model-improvement.md).

The decision classifier is **HistGradientBoosting**, not Random Forest. The
person-feature transformer and temporal CNN experiments are documented in the
[approach comparison](play-approach-comparison-2026-11-07.md). A simple MLP,
AdaBoost, Random Forest and Extra Trees have not yet received the same complete
game-grouped comparison; the current default is the strongest validated pipeline,
not a claim that every classifier family has been exhausted. Any replacement
must improve event precision while retaining at least 95% per-game recall, with
thresholds selected on development games before held-out evaluation. Compare the
same features and folds, and report clip boundaries and complete pipeline time
alongside classifier inference time.

Studio defaults to **Default model · automatic GPU / CPU** when an active model
or bundled default is available. Explicit saved profile choices remain respected.
The processing worker selects a compatible NVIDIA reference runtime automatically
or uses the pack's CPU graph. Saved fields, reviews and game configuration are not
rewritten by installing or activating a default model.

Standard reference builds include the versioned project-trained decision models
from `models/catalog.json`. Startup prepares them in a background task without
downloading generic detector weights. First detection fetches the pinned official
YOLO26n weights and caches them. CPU OpenVINO/ONNX exports are generated once in
the cache. Version 0.5.0 retains the base decision trees and adds visual-only boundary-window
scoring. The frozen four-game result is 73.48% for both cuts within ±1 second,
92.05% play precision and 98.78% recall. This is repeated confirmation, not a new
blind test or a 95% timing guarantee. See [refiner validation](visual-refiner-2026-11-13.md). See [model inventory](model-inventory.md).

Bundled defaults are immutable installed versions. Automatic upgrades apply only
to a previously managed bundled default; manual model selections and rollbacks
are preserved. Pointer snapshots prevent a concurrent model choice being replaced
by a background default update. Damaged archives/manifests fail validation.

## Building a standalone with the selected pipeline

The ordinary build includes only our decision models and matching configuration:

```bash
python scripts/build-installers.py --runtime-profile reference
```

An optional `--baseline-pack` plus `--baseline-sha256` override must also be a
verified schema-3 decision-only pack. Build inputs containing pretrained detector
graphs are rejected. Legacy CPU packs remain importable at runtime. The reduced
inference-only build needs a separately imported ready-to-run CPU pack.
Training checkpoints, GT, footage and arbitrary model-directory contents are not
included. Source archives and wheels use an explicit model-file allowlist.

An explicitly supplied FFmpeg is copied into `_scout_media`; PyInstaller collects
its dynamic dependencies. This allows the validated CUDA decoding/resizing path
in the reference build. Without that build option, the existing imageio FFmpeg
fallback remains. CUDA inference and CUDA video decoding are separate features.
Existing local Linux X11 cursor shims are included when present; no dependency is
downloaded by the spec. The installed smoke checks use the selected media engine
and an isolated model store, never the user's model store or footage.

## Local launcher selection

`start-studio.sh` can use an explicitly configured standalone executable from the
ignored `.scout-studio-runtime` path file. It preserves the workspace and quoted
arguments. If that selected executable is missing, it reports the problem instead
of silently using another processing environment. Removing the path file restores
the ordinary source-environment fallback. Existing game data is independent of
this launcher selection.

## Local deployment validation

The approved `scout-reviewed-precision@0.3.0` pack and its matching defaults are
now active for this user's normal Studio launch. The Applications entry and
project launcher select the newly frozen reference build. The previous desktop
entry and detection-profile preference were backed up before changing them.
At that deployment, the original PKL, game configuration, field polygons and review files were not
replaced. No personal footage was processed during deployment.

Validation passed on the local Linux machine:

- 325-test suite, with two native-platform skips; source/wheel checks passed.
- Independent default-activation review, including a reproduced pointer race,
  corrected snapshot/CAS logic and regression checks.
- Actual standalone native Qt/X11, synthetic playback/export, CLI, worker,
  OpenVINO and bundled-default preparation smoke checks.
- Default Studio job queue to the actual frozen worker, using synthetic video:
  automatic CUDA and CUDA-disabled CPU fallback both completed successfully.
- Normal launcher dispatch lists the selected model from the actual user store.
- Nine-game original integrity checks still pass: 109 snapshotted state-file hashes
  and 41 original video size/mtime identities.

The standalone carries the tested FFmpeg and approved model archive. Source
weights remain private; no model/release upload or release tag was performed.
This validates the local Linux deployment. Windows installer execution and GPU
hardware still require native validation; no new Windows installer was built here.

## Decision-only packaging update

The current source catalogue carries the approved decision head and selected
Person Transformer. Generic detector weights and CPU exports are excluded from
standard installer payloads. The historical deployment evidence above describes
0.3.0; it does not establish Windows/macOS execution of this packaging update.
