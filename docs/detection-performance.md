# Detector performance and troubleshooting

The managed default is YOLO26n1280 at 2.5 Hz, not the old YOLO11m1280/5Hz CPU
baseline. [Current model status](model-status.md) reports complete pipeline speed;
[historical comparisons](research-ledger.md#historical-detector-measurements--5-november)
retain all nine detector timings. Cached detections do no inference, so a cache
hit need not show GPU activity.

## Automatic acceleration

First-launch setup selects compatible NVIDIA CUDA packages on Windows/Linux,
otherwise CPU; macOS uses its native CPU/Metal engine. A worker probe validates
compatibility and keeps CPU fallback. The original Windows build carried CPU-only
Torch; changing system Python could not give that frozen application CUDA.
Use a current installer and working driver. GPU-less CI verifies packaging and
fallback, not physical Windows GPU speed. CUDA detector inference and FFmpeg
GPU decode are independent capabilities; see [installation](installation.md).

The managed CPU path uses the pack's explicit graph backend. Graph conversion
occurs in private cache staging with checked weight identity, not beside original
weights. Warm exported inference needs no Torch import. Backend/cache identities
remain separate; CPU FP32 uses batch 1, unlike the GPU batching choice.
`inference_threads=0` lets the engine choose, and positive values limit threads.
Quantized/alternative detector layouts require explicit compatibility and quality
checks; INT8 is not an automatically safe optimization.

## Controlled source comparisons

The existing experimental CPU profile and custom detector settings remain tools
for comparison, not substitutes for the managed default's quality evidence.
Python source installs need `[inference]` and `[reference]` for first conversion.
For example, an explicit historical comparator is:

```bash
scout --source-dir '/path/to/Game' detect --model yolo11s.pt \\
  --imgsz 1280 --detector-backend openvino --device cpu --batch 1
```

Use `onnx` for ONNXRuntime. Never change saved production configuration/cache
merely to validate another backend. The early ONNX/OpenVINO adapters matched
reference counts/order on eight frames, with maximum coordinate delta 0.000244 px
and confidence delta <0.000003. That establishes export consistency on those
inputs, not held-out play accuracy.

## Reproducing local measurements

Use `scripts/benchmark-detectors.py --help` and a separate directory of copied
representative frames, including distant players/crowded formations. Hold
confidence, sampling, resolution and selected frames constant when comparing
runtimes. Record hardware, precision, batch, threads, cold/export costs and warm
medians; explicitly select CPU when measuring CPU fallback.

Then score full play intervals against reviewed GT using the
[evaluation protocol](model-improvement.md). Detector box agreement and vendor
COCO scores cannot establish play recall. Lowering resolution lost visible
players in the recorded 960 comparison; retain 1280 until another full pipeline
earns a better quality/speed tradeoff. New results belong in fresh experiment
directories with versions, hashes and outputs, not original Scout folders.
