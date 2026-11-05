# Local detector comparison — 2026-11-05

Nine pretrained detectors were measured on the same eight copied 1920×1080 game
frames. All 45 model/runtime/batch combinations completed. The largest practical
CPU improvement comes from a smaller model; changing inference engines alone
does not make the current medium model realtime.

## Environment and isolation

- Intel Core i7-11700F, 8 cores / 16 threads; NVIDIA RTX 3060 Ti, 8 GB.
- Linux x86-64; NVIDIA driver 595.99.02. This is the Windows user's hardware,
  running Linux; these measurements do not validate Windows performance.
- Python 3.12.3, PyTorch 2.14.1+cu130, Ultralytics 8.4.183,
  ONNX Runtime 1.31.1, OpenVINO 2026.4.1.
- Separate temporary virtual environment and workspace; an 18-second video
  excerpt and eight frames were copied there. Weights were downloaded into
  that workspace. Existing game `Scout` state, fields, reviews, configuration,
  exports and detection caches were not read or written.
- Ultralytics, matplotlib and Torch configuration/cache paths were isolated;
  package autoinstall was disabled. The isolated virtual environment read the
  development environment's existing dependencies and held additional packages
  locally. It did not duplicate the large PyTorch installation.

## Warm inference results

Milliseconds per frame, lower is faster. CPU runs use FP32 weights; GPU runs
use PyTorch FP16. All inputs use `imgsz=1280`, person class only, confidence
0.25, rectangular letterboxing and an actual tensor size of 736×1280. The
YOLO26 models use the ordinary default detection head, not an explicitly
selected NMS-free head.

| Model | CPU torch, batch 1 | CPU torch, batch 8 | CPU ONNX, batch 1 | CPU OpenVINO, batch 1 | GPU torch, batch 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 62.0 | 65.2 | 92.6 | 55.6 | 8.1 |
| YOLOv8s | 148.5 | 162.3 | 179.4 | 134.5 | 11.3 |
| YOLOv8m | 344.0 | 351.6 | 343.2 | 305.6 | 19.1 |
| YOLO11n | 60.3 | 65.9 | 75.2 | 46.9 | 8.1 |
| YOLO11s | 142.0 | 145.0 | 135.4 | 109.4 | 10.8 |
| YOLO11m — current | 349.2 | 368.0 | 502.5 | 288.3 | 18.1 |
| YOLO26n | 56.0 | 61.0 | 40.3 | 39.0 | 7.3 |
| YOLO26s | 134.4 | 134.1 | 107.0 | 98.1 | 9.7 |
| YOLO26m | 325.9 | 353.9 | 307.1 | 275.1 | 16.4 |

Each combination runs in a fresh subprocess. A full pass warms up every frame
before three timed passes over all eight frames. Values are the median per-frame
time across those passes, including preprocessing and postprocessing/NMS. CUDA
is synchronized around timings. Exports and cold startup are excluded. CPU
torch reports eight threads; OpenVINO manages its own inference threads.
All OpenVINO logs explicitly report CPU execution. The utility also forces
`intel:CPU` and asserts the compiled execution device, preventing AUTO from
silently selecting another accelerator on other machines.

At five sampled frames per video second, 200 ms/frame is the inference-only
realtime threshold. YOLO26s/OpenVINO is about 3.75× faster than the current
YOLO11m/torch/batch-eight inference path. YOLO11s/torch/batch-one is about 2.59×
faster without adding a runtime. Those ratios do not include video decoding.

ONNX is not a universal speed improvement here: it is slower for several models,
especially YOLO11m. CPU batch-eight is usually slightly slower than batch-one,
so the GPU batch setting should not be assumed optimal for CPU.

## Complete CPU detection stage

The same copied 18.01-second excerpt was processed at five sampled frames per
second (90 frames). Each case used a fresh subprocess and a distinct cache
directory inside the temporary workspace. FFmpeg decoding was explicitly forced
to CPU, even though this machine has a GPU. The timer includes model loading,
decoder work, preprocessing, inference, postprocessing and saving detections.
Python/package imports, initial model downloads and graph export are excluded.
OpenVINO was explicitly forced to CPU. No play segmentation or review generation
was run; these are full **detection-stage** measurements, not the whole app workflow.

| CPU detector | Batch | Seconds for 18.01s footage | Realtime factor | Speed vs current |
| --- | ---: | ---: | ---: | ---: |
| YOLO11m / PyTorch — current | 8 | 33.16 | 0.54× | 1.00× |
| YOLO11s / PyTorch | 8 | 14.98 | 1.20× | 2.21× |
| YOLO11s / PyTorch | 1 | 13.92 | 1.29× | 2.38× |
| YOLO26s / PyTorch | 1 | 14.06 | 1.28× | 2.36× |
| YOLO11s / OpenVINO | 1 | 12.12 | 1.49× | 2.74× |
| YOLO26s / OpenVINO | 1 | 12.19 | 1.48× | 2.72× |
| YOLO26n / OpenVINO | 1 | 6.85 | 2.63× | 4.84× |

Each detection-stage case was run once, including cold model initialization;
these figures have more uncertainty than the three-pass warm inference timings.
Initial graph export is an additional one-time cost for the exported runtimes.
The user's Windows result was 0.3× for the current model. This Linux excerpt
does not reproduce that exact result, so neither the absolute numbers nor a
simple ratio applied to 0.3× should be treated as a Windows guarantee.

**CPU recommendation:** evaluate YOLO11s at 1280 and batch one first. It achieves
1.29× here with the existing PyTorch runtime, and 1.49× with OpenVINO. YOLO26s
does not deliver a material detection-stage advantage on this short excerpt.
OpenVINO remains an experimental benchmark path, not a bundled automatic Scout
backend. Its integration needs safe reusable exports, backend-specific detection
cache identity and native installer checks. Keep the current default until
quality is measured across more scenes and corrected plays.

## Detection agreement

These are unlabeled frames from one short scene, including distant players and
the sideline. There is no ground-truth accuracy or play recall measurement.
For a rough comparison, candidate boxes were greedily matched one-to-one with
the current YOLO11m CPU batch-one boxes at IoU ≥0.5. A separate, temporary
field polygon was drawn for this comparison; no saved fields were accessed.
Boxes are selected by bottom-center inside that polygon.

| Candidate | Field boxes matched / YOLO11m field boxes | Reference coverage |
| --- | ---: | ---: |
| YOLOv8n | 98 / 121 | 81.0% |
| YOLOv8s | 99 / 121 | 81.8% |
| YOLOv8m | 111 / 121 | 91.7% |
| YOLO11n | 90 / 121 | 74.4% |
| YOLO11s | 107 / 121 | 88.4% |
| YOLO11m | 121 / 121 | 100% by definition |
| YOLO26n | 83 / 121 | 68.6% |
| YOLO26s | 103 / 121 | 85.1% |
| YOLO26m | 108 / 121 | 89.3% |

Reference coverage is agreement with another detector, **not accuracy**. The
reference can miss people or produce false positives; candidates also add boxes
that the reference does not contain. A box mismatch need not mean a missed
player. This comparison cannot justify silently replacing the default detector.
Nano models are attractive for speed but require a stronger quality assessment.

The temporary polygon's normalized vertices are `(0, .535), (.61, .52),
(1, .80), (0, 1)`, using the same 1920×1080 frame dimensions for every model.

## Reproduction

Use [scripts/benchmark-detectors.py](../scripts/benchmark-detectors.py) with an
isolated directory of eight copied PNG frames and a fresh output directory.
The JSON output includes per-frame boxes, input shapes, timing spread, export
times and dependency versions. See [the procedure](detection-performance.md#reproducing-local-measurements).
The numeric findings above are committed without footage, images, weights or
personal library paths. The original model, resolution and saved game settings
remain unchanged.
