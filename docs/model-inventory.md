# Model inventory and distribution

Only two trained approaches are retained: the approved YOLO + HGB pipeline and
our experimental Person Transformer. `models/catalog.json` lists the exact
files shipped in source archives, wheels and standard reference installers.

| Component | Purpose | Weight origin | Distribution |
| --- | --- | --- | --- |
| YOLO26n | Detect people in sampled frames | Official Ultralytics pretrained COCO weights; not trained on these games | Pinned official asset, downloaded on first detection and cached |
| Tracking, motion and formation features | Convert person detections into temporal play evidence | Deterministic Scout code | Application source |
| HGB `classifier.json` | Decide which intervals contain plays | Trained and calibrated on the reviewed development games | `models/scout-reviewed-precision/0.4.0/` |
| Person Transformer `head.onnx` | Alternative temporal decision model over the same person features | Best tested project-trained Transformer | `models/person-transformer/1.0.0/` |

The HGB classifier bytes in 0.4.0 are identical to accepted 0.3.0. This version
changes distribution and lazy detector preparation, not training or accuracy.
The recipe binds detector, sampling, features, classifier and interval settings;
copying the JSON into another pipeline does not reproduce the measured results.

## Evidence and retained alternatives

See [model improvement](model-improvement.md) and
[approach comparison](play-approach-comparison-2026-11-07.md) for splits,
interval matching, timing and limitations. Games held out from fitting are kept
separate from development games; these are measured results, not guarantees.

| Approach | Validation precision | Validation recall | Decision |
| --- | ---: | ---: | --- |
| Approved HGB, GPU pipeline | 92.05% | 98.78% | Default |
| Approved HGB, CPU pipeline | 92.53% | 98.17% | Default CPU fallback |
| Person Transformer | 87.53% | 98.48% | Selectable experiment |
| S3D temporal head | 40.24% | 91.16% | Retire model binaries |
| MobileNet temporal head | 35.61% | 58.84% | Retire model binaries |

On the separately evaluated NT17-SAI game, the approved GPU pipeline measured
95.68% precision and 98.52% recall (133/135 plays). Full PAN timing measured
15.86× realtime on GPU and 6.59× on CPU on this machine. The four-game aggregate
precision remains below 95%; do not describe this as universal 95% precision.

## First use, storage and compatibility

Standard installers carry our decision weights only. YOLO26n comes from the
[official pinned release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt),
with SHA-256 verification before loading. CPU OpenVINO or ONNX detector graphs
are generated once in the application cache and reused. First detection needs
internet and the reference/export runtime; opening Studio does not download
weights. The reduced inference-only build needs an imported ready-to-run CPU pack.

Installed model versions live in the OS application-data `model-packs` directory,
and alternative recipes in `play-pipelines`. Existing manual choices and rollback
versions remain preserved. On this Linux machine the parent directory is
`~/.local/share/Scout/Scout Studio/`. Cached generic weights are separate from
these project-trained decision models.

The legacy root `models/play.pkl` is retired. A missing standard legacy default
is redirected to the approved model; custom model paths remain explicit. Its
old bytes remain recoverable from Git history. Benchmark reports, reviewed GT,
feature arrays, footage and user review files are not removed with retired models.

## License notices

Project-trained decision weights reference the central project `LICENSE`; they
do not duplicate its full text in each model directory. Separately obtained
pretrained weights and runtime dependencies retain their upstream terms. Being
open source and noncommercial does not remove those terms.

## Cleanup record — 2026-11-08

Removed 46 inventoried retired model/recipe files (about 278 MiB logical bytes)
from the isolated approach study, its installed S3D/MobileNet alternatives and
the identified S3D download cache. A size/SHA-256 receipt remains beside the
study's reports as `retired-models-2026-11-08.json`. Best Person Transformer
inference weights and its training state were retained. Logical bytes can differ
from physical space freed when model copies are hardlinked.

## Packaging validation — 2026-11-08

- Full check: 376 tests passed, with two native-platform skips; agent configuration
  validation and CLI help also passed.
- Source/wheel checks passed, including exact release contents and installed CLI.
- Catalogue/installer regressions passed: nested-link/checksum rejection, macOS
  distribution-root links and rejection of detector-containing installer overrides.
  Runtime imports of legacy CPU packs remain supported.
- Independent correction review found no unresolved high/medium findings.
- Rebuilt Linux standalone passed native Qt/X11 startup, playback, synthetic
  export, CLI, worker, ONNX/OpenVINO and isolated default bootstrap.
- Actual frozen CPU detection completed on a synthetic two-second video with a
  fresh OpenVINO export (verified pretrained checkpoint preseeded in a private
  cache). This checks export/runtime compatibility, not game accuracy or speed.
- A prior four-real-frame read-only check found zero box/confidence differences
  between the fresh CPU export and the approved CPU graph. It does not replace
  the historical full-game accuracy/speed evaluations.

Windows/macOS execution of this packaging change requires fresh hosted checks.
