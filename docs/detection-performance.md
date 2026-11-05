# Detector performance

Scout defaults to `yolo11m.pt`, `imgsz=1280`, `sample_fps=5`, `batch=8`, and
`device=auto`. Automatic selection uses NVIDIA CUDA when PyTorch reports it
available, otherwise CPU. Uncached detection logs the selected model, device,
resolution and batch size, including the reason for automatic CPU fallback.
Cached detections do not run inference and therefore need no GPU activity.

## Windows installation finding

The original standalone Windows build explicitly installed CPU-only PyTorch.
Automatic device selection cannot enable CUDA in that build, even with an NVIDIA
GPU and driver installed. New builds use the official CUDA 12.8 PyTorch wheel
index and assert CUDA build support before packaging. CPU fallback remains
available. The frozen Windows build, GPU drivers and actual GPU inference still
require Windows hardware validation; a CI runner without a GPU verifies only
that CUDA support is bundled. Updating system Python does not update the frozen app.

## Faster model candidates

Published manufacturer measurements checked on 2026-11-05:

| Detector | COCO box mAP 50–95 | CPU ONNX ms/image | Relative CPU speed vs YOLO11m |
| --- | ---: | ---: | ---: |
| YOLO11m (current) | 51.5 | 183.2 | 1.00x |
| YOLO11s | 47.0 | 90.0 | 2.04x |
| YOLO11n | 39.5 | 56.1 | 3.27x |
| YOLO26s | 48.6 | 87.2 | 2.10x |
| YOLO26n | 40.9 | 38.9 | 4.71x |
| YOLO26m | 53.1 | 220.0 | 0.83x |

Sources: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).
These are vendor reference figures at **640 pixels using ONNX**, not Scout
measurements at 1280 using PyTorch, and are not end-to-end video throughput.
YOLO26 speed uses its NMS-free head; the reported accuracy column above is its
ordinary head (NMS-free scores are 47.8 for small and 40.1 for nano).
Ratios are reference latency ratios, not promised Scout speedups.

Start by comparing **YOLO11s at 1280** against the current medium model. It works
with the existing YOLO11 dependency range and avoids changing resolution and
model together. YOLO26s is another candidate with newer Ultralytics, although its
published CPU advantage over YOLO11s is small. Nano models trade much more
accuracy for speed. YOLO26m does not offer a CPU speed improvement in this table.
None of these COCO scores establishes similar recall for distant football
players or similar play segmentation accuracy. See the
[local nine-model comparison](detector-benchmark-2026-11-05.md) for measured
performance on the i7-11700F and RTX 3060 Ti.

For CPU optimization while keeping the same weights, evaluate FP32 ONNX Runtime
and, on Intel hardware, OpenVINO. Ultralytics documents
[OpenVINO acceleration](https://docs.ultralytics.com/integrations/openvino/),
but neither runtime is currently bundled by Scout's standard installer.
They need explicit dependency, packaging, batch-shape and cache validation before
being offered as automatic backends. INT8 additionally needs representative
calibration and accuracy checks.

## Safe comparison procedure

Use a separate temporary workspace with a short representative excerpt, including
far-side players and crowded formations. Keep production footage, configuration,
reviews and caches unchanged. For each candidate, hold resolution, confidence,
sample rate and frame selection constant. Warm up inference, then measure model
latency and complete decode/preprocessing/inference time separately on the target
Windows CPU and GPU. Record hardware, runtime version, precision and batch size.
Compare person recall against labeled frames and final play proposals against
corrected review ranges; agreement with the existing model alone is not accuracy.

Do not lower the default resolution or replace the default detector based only
on the vendor numbers. Scout's configuration documents that reducing resolution
can lose players on the far side. The local comparison measures inference speed
and agreement on a short scene; equivalent play-detection quality remains unverified.

## Reproducing local measurements

`scripts/benchmark-detectors.py` compares YOLOv8, YOLO11 and YOLO26 nano/small/medium
models on the same copied PNG frames. Use a separate virtual environment containing
Ultralytics, PyTorch, OpenCV, NumPy, ONNX, ONNX Runtime and OpenVINO. The script
does not install missing packages. Choose only installed runtimes with `--runtimes`.

Copy representative frames into an isolated directory; the default batch-eight
measurements need a frame count divisible by eight, with matching image dimensions.
Pass a new or empty output directory:

```bash
python scripts/benchmark-detectors.py \
  --frames /tmp/scout-comparison/frames \
  --workspace /tmp/scout-comparison/results --gpu
```

Omit `--gpu` for a CPU-only system. Model downloads, export graphs, package settings,
logs and incremental `report.json` stay inside the output directory. Each model,
runtime and batch combination runs in a fresh subprocess. Timings include
preprocessing, inference and NMS after a full warmup pass. The median covers three
passes over every frame; GPU timing synchronizes CUDA. CPU exports use FP32
weights and dynamic rectangular inputs; GPU PyTorch uses FP16. The report records
actual input shapes and predictions, so exported model agreement can be checked.

These are inference measurements. Decoding, cold startup and initial export are
excluded; the inferred realtime factor assumes five sampled frames per video
second. Unlabeled frame agreement does not establish accuracy or play recall.
