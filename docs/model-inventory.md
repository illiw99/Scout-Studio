# Model inventory and distribution

Two approaches are retained in `models/catalog.json`: approved YOLO + HGB and
the experimental Person Transformer. The catalogue is the exact source/wheel/
installer allowlist, not a scan of arbitrary model-directory contents.

| Component | Purpose | Weight origin / delivery |
| --- | --- | --- |
| YOLO26n | People in sampled frames | Official Ultralytics COCO pretrained weights; not trained on these films. Pinned official release/SHA-256, downloaded at first detection |
| Tracking, motion, formation and temporal features | Play evidence from person boxes | Deterministic application code |
| `scout-reviewed-precision/0.5.0/classifier.json` | HGB play decisions + two boundary heads | Project-trained/calibrated numeric weights, bundled with matching configuration |
| `scout-reviewed-precision/0.4.0/` | Prior HGB recipe | Retained immutable rollback and Transformer feature dependency; base classifier bytes match accepted0.3.0 |
| `person-transformer/1.0.0/head.onnx` | Alternative temporal decision head | Project-trained best retained Transformer, selectable experiment |

The 0.5.0 boundary heads require no audio or new visual backbone. Classifier JSON
is sufficient only with its bound detector/sampling/features/postprocessing.
[Current results](model-status.md) and the [rejection ledger](research-ledger.md)
explain why HGB remains default. A result near95% on one game is not95% aggregate
precision. The legacy `models/play.pkl` is retired; custom trusted PKL paths
remain explicit, while the missing standard legacy default redirects to the
approved pack. Old bytes remain recoverable from Git.

## Storage and first use

Generic detector weights come from the [official pinned asset](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt).
CPU OpenVINO/ONNX graphs are generated once in the cache, not packaged in the
installer. First export requires the processing/reference dependencies. The
explicit reduced inference-only developer profile instead needs an imported
ready CPU pack. [Engine setup](runtime-delivery.md) describes private upstream
Python/packages; source installations use their installed environment.

Versioned models live in the OS application-data `model-packs` directory;
alternative recipes in `play-pipelines`, generic weights in a separate cache.
On Linux the application-data parent is `~/.local/share/Scout/Scout Studio/`.
Manual selections and rollback versions are preserved.

## Licensing and retention

Project-trained weights reference the central project `LICENSE`. Third-party
weights/dependencies retain their upstream terms even for noncommercial open
source use. Retired S3D/MobileNet binaries and the previous cleanup receipt are
recorded in [maintenance history](maintenance-history.md). Research reports,
reviewed GT, footage, caches and trained model evidence were not removed by the
documentation cleanup.
