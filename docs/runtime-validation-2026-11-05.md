# Isolated runtime validation — 5 November 2026

The pinned Linux CPU dependency set installed successfully in a new temporary
Python3.12 environment. Real PyInstaller/`dpkg-deb` builds produced both reference
and experimental inference-only installers. Each package was extracted into a
temporary directory and its frozen executable was tested there. No package was
installed into `/opt`, no uninstall ran, and no existing game `Scout/` folder was
read or changed. Windows/macOS installers, signing and upgrades remain unverified.

## Bundle measurements

| Linux x86-64 profile | Installer | Extracted regular files |
| --- | ---: | ---: |
| Reference | 439.4 MiB | 1753.8 MiB |
| Experimental inference-only | 228.9 MiB | 807.2 MiB |

Exact byte counts are 460755632/1839025625 for reference and
240000668/846368792 for inference. Extracted size sums regular file lengths,
excluding symlinks; disk allocation depends on the filesystem. No detector weights
or play classifier are included. The reference build is commit `34ec4e7`; the
inference build adds the reviewed runtime-profile implementation. Both use the
same Linux dependency lock. This compares these local prototypes, not released
Windows installer sizes.

Both frozen smoke runs passed FFmpeg decode, synthetic MP4 export, CLI dispatch,
versioned worker execution, ONNX and OpenVINO CPU inference, native Qt window and
offscreen playback. Native application-data paths and QSettings locations matched
seeded temporary XDG paths. Library index, model-pack pointer, preferences and
synthetic game sentinels remained byte-identical. Offscreen playback is not proof
of desktop display, audio, codecs or hardware acceleration on another machine.

The inference smoke also established that Torch, TorchVision, Ultralytics,
scikit-learn and matplotlib are absent. It consumes exported graphs and numeric
classifiers. This is an explicit build option, not the production default:

```bash
python scripts/build-installers.py --runtime-profile inference
```

Its separate outputs are `build/installer-inference` and
`dist/installers-inference`. A tested compatible model pack must be imported
before detection. Checkpoint export and training need the source/reference tools;
the native selector disables checkpoint-based CPU export for this profile.
The candidate manifest and installed test bind the runtime profile. Default CI
continues building the reference profile pending quality and offline-baseline
approval.

## Actual classifier and pack checks

A copy of the current trusted local classifier was exported to numeric JSON. The
original file's hash was unchanged. The export contains 198 windowed features,
sampling 5 fps, and 743563 bytes. Across 24968 threshold/adjacent-float/NaN probes,
maximum probability error against scikit-learn was`1.1102230246251565e-16`.

On feature vectors from three existing temporary 90-frame detection runs, maximum
classifier error was at most`5.551115123125783e-17`. Both classifiers returned no
play intervals under the validation's full-image mask. Empty intervals do not
validate detection of real plays or establish boundary quality.

Local review-pending packs were built from allowlisted inputs, imported, probed
and activated: reference YOLO11m/ONNX and smaller YOLO11s/OpenVINO, each with the
numeric classifier. They contain no pickle, training provenance or footage.
They are temporary validation artifacts, not approved redistribution inputs or
an offline production baseline.

The inference-only frozen executable imported and activated the CPU pack, then
processed a new copy of the 18.01-second excerpt through a versioned worker job:
90 sampled frames, pack identity recorded in the NPZ cache, verified successful
worker result and a new review MLT. It took 14.82 seconds including worker startup
and pipeline work. OpenVINO was configured for CPU, CUDA visibility disabled;
all state went to a new temporary game. This is a functional integration timing,
not a repeated speed benchmark. Use the [controlled detector comparison](detector-benchmark-2026-11-05.md)
for model/runtime speed results.

## Remaining acceptance gates

- Labeled held-out games, positive/negative plays, joins, fractional frame rates,
  short/long plays, boundary errors and manual correction effort.
- Representative startup/peak-runtime-memory/stage profiles; build-process memory
  and smoke duration do not measure these.
- Approved redistribution inputs and publisher-authenticated model channel.
- Native Windows/macOS builds, real installed upgrade/uninstall, shortcut launches,
  signing/notarization and hosted promotion of exact tested bytes.

The experimental CPU option preserves 1280 px sampling resolution. Smaller models
can miss distant players; the current quality default remains unchanged.
