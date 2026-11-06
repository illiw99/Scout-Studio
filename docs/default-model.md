# Default Scout Studio detection profile

The selected local profile is `scout-reviewed-precision@0.3.0`: YOLO26n at 1280,
2.5 samples/s, adaptive fields, time-scaled tracking, derived play features and
wide temporal context. The profile binds its classifier, detector, sampler and
postprocessing together; replacing only an old PKL is insufficient. Its operating
point prioritizes precision while keeping recall above 95% in every measured
validation game. See [measurements and limitations](model-improvement.md).

Studio defaults to **Default model · automatic GPU / CPU** when an active model
or bundled default is available. Explicit saved profile choices remain respected.
The processing worker selects a compatible NVIDIA reference runtime automatically
or uses the pack's CPU graph. Saved fields, reviews and game configuration are not
rewritten by installing or activating a default model.

A standalone build can carry an explicitly approved offline pack. First startup
prepares it in a background task; the processing worker also ensures it is ready
before a queued job begins. No model probing runs on the Qt UI thread. The CPU
model is included, so it needs no model download. GPU reference weights use the
existing pinned official-asset/hash verification and CPU fallback behavior.

Bundled defaults are immutable installed versions. Automatic upgrades apply only
to a previously managed bundled default; manual model selections and rollbacks
are preserved. Pointer snapshots prevent a concurrent model choice being replaced
by a background default update. Damaged archives/manifests fail validation.

## Building a standalone with the selected pipeline

Supply a private approved pack and its independently recorded archive hash:

```bash
python scripts/build-installers.py --runtime-profile reference \
  --baseline-pack /private/reviewed-model.zip --baseline-sha256 EXPECTED_SHA256 \
  --ffmpeg-binary /path/to/tested/ffmpeg \
  --work-dir /private/new-build --output-dir /private/new-installers
```

The baseline is validated/probed and copied into `_scout_baseline` with a receipt.
Only `model-pack.zip` and `receipt.json` are included. Private GT, footage, training
arrays and arbitrary model-directory contents are never included. No trained
weights enter Git. Build environments need the matching CPU runtime; for example,
macOS needs a compatible ONNX pack instead of an OpenVINO-only pack.

An explicitly supplied FFmpeg is copied into `_scout_media`; PyInstaller collects
its dynamic dependencies. This allows the validated CUDA decoding/resizing path
in the reference build. Without that build option, the existing imageio FFmpeg
fallback remains. CUDA inference and CUDA video decoding are separate features.
Existing local Linux X11 cursor shims are included when present; no dependency is
downloaded by the spec. The installed smoke checks use the selected media engine
and an isolated model store, never the user's model store or footage.

Builds without an approved baseline remain supported for development/review;
they do not automatically acquire private weights. Public distribution is a
separate release action.

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
The original PKL, game configuration, field polygons and review files were not
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
