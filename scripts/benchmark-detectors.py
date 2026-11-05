#!/usr/bin/env python3
"""Compare detector inference in an isolated workspace, never in a game folder.

Input is an explicit directory of copied frame images. Each measurement gets a
fresh subprocess; weights and exports stay in the output workspace. Results are
speed and detection agreement, not labeled accuracy or full-pipeline throughput.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(settings):
    # Package configuration and downloads must remain inside this workspace.
    root = Path(settings['workspace'])
    os.chdir(root)
    os.environ['YOLO_CONFIG_DIR'] = str(root / 'ultralytics-config')
    os.environ['MPLCONFIGDIR'] = str(root / 'matplotlib-config')
    os.environ['TORCH_HOME'] = str(root / 'torch-cache')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['DO_NOT_TRACK'] = '1'
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    frames = [cv2.imread(str(path)) for path in sorted(Path(settings['frames']).glob('*.png'))]
    if not frames or any(frame is None for frame in frames):
        raise ValueError('Frame directory must contain readable PNG images')
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError('Use frames of the same dimensions for a comparable rectangular input')
    batch = settings['batch']
    if len(frames) % batch:
        raise ValueError('Frame count must be a multiple of the batch size')
    groups = [frames[i:i + batch] for i in range(0, len(frames), batch)]
    runtime, device = settings['runtime'], settings['device']
    precision = 'fp16' if device != 'cpu' else 'fp32'
    precision_args = ({'quantize': 16 if precision == 'fp16' else None}
                      if 'quantize' in DEFAULT_CFG_DICT else {'half': precision == 'fp16'})
    model = YOLO(settings['model'])
    export_seconds = 0.
    if runtime != 'torch':
        start = time.perf_counter()
        arguments = dict(format=runtime, imgsz=settings['imgsz'], dynamic=True,
                         batch=1, device='cpu', **({'quantize': None}
                         if 'quantize' in DEFAULT_CFG_DICT else {'half': False}))
        if runtime == 'onnx':
            arguments.update(simplify=False, opset=17)
        exported = model.export(**arguments)
        export_seconds = time.perf_counter() - start
        model = YOLO(exported, task='detect')
    # OpenVINO can otherwise choose AUTO even when Ultralytics received 'cpu'.
    predict_device = 'intel:CPU' if runtime == 'openvino' else device
    options = dict(imgsz=settings['imgsz'], conf=.25, classes=[0], device=predict_device,
                   verbose=False, rect=True, **precision_args)

    def synchronize():
        if device != 'cpu':
            torch.cuda.synchronize()

    predictions = []
    # Warm up every frame and shape before timing. Keep one set of boxes for
    # comparisons without repeating tensor transfers inside the timed section.
    for group in groups:
        results = model.predict(group, **options)
        synchronize()
        predictions.extend([np.column_stack((r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy())).tolist() for r in results])
    shape = list(model.predictor.preprocess(groups[0]).shape)
    sweeps, batches = [], []
    for _ in range(settings['repeats']):
        synchronize()
        start = time.perf_counter()
        for group in groups:
            before = time.perf_counter()
            model.predict(group, **options)
            synchronize()
            batches.append((time.perf_counter() - before) * 1000 / len(group))
        sweeps.append((time.perf_counter() - start) * 1000 / len(frames))
    median = statistics.median(sweeps)
    result = dict(model=settings['model'], runtime=runtime, device=device, precision=precision,
                  batch=batch, imgsz=settings['imgsz'], input_shape=shape,
                  frames=len(frames), repeats=settings['repeats'], ms_per_frame=median,
                  min_ms_per_frame=min(sweeps), max_ms_per_frame=max(sweeps),
                  p95_batch_ms_per_frame=float(np.percentile(batches, 95)),
                  inference_realtime_at_5fps=1000 / (median * 5),
                  export_seconds=export_seconds, predictions=predictions,
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                ultralytics=ultralytics.__version__),
                  torch_threads=torch.get_num_threads())
    if runtime == 'onnx':
        import onnxruntime
        result['versions']['onnxruntime'] = onnxruntime.__version__
    elif runtime == 'openvino':
        import openvino
        result['versions']['openvino'] = openvino.__version__
        compiled = model.predictor.model.ov_compiled_model
        devices = list(compiled.get_property('EXECUTION_DEVICES'))
        if not devices or any(not item.startswith('CPU') for item in devices):
            raise RuntimeError(f'Expected OpenVINO CPU execution, got {devices}')
        result['execution_devices'] = devices
        result['execution_precision'] = str(compiled.get_property('INFERENCE_PRECISION_HINT'))
    Path(settings['result']).write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, help='isolated copied PNG frames (read only)')
    parser.add_argument('--workspace', type=Path, help='new, empty benchmark output directory')
    parser.add_argument('--models', nargs='+', default=[f'{family}{size}.pt' for family in
                        ('yolov8', 'yolo11', 'yolo26') for size in ('n', 's', 'm')])
    parser.add_argument('--runtimes', nargs='+', choices=('torch', 'onnx', 'openvino'),
                        default=['torch', 'onnx', 'openvino'])
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--repeats', type=int, default=3, help='timed passes over every frame')
    parser.add_argument('--gpu', action='store_true', help='also benchmark CUDA FP16 batch 8')
    parser.add_argument('--worker-settings', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_settings:
        worker(json.loads(args.worker_settings.read_text()))
        return
    if not args.frames or not args.workspace or args.imgsz <= 0 or args.repeats <= 0:
        parser.error('--frames, --workspace, positive --imgsz and --repeats are required')
    frames, root = args.frames.resolve(), args.workspace.resolve()
    if not frames.is_dir() or not list(frames.glob('*.png')):
        parser.error('--frames must contain copied PNG images')
    if root.exists() and any(root.iterdir()):
        parser.error('--workspace must be new or empty; existing directories are never reused')
    if any(Path(name).name != name or not name.startswith(('yolov8', 'yolo11', 'yolo26'))
           or not name.endswith('.pt') for name in args.models):
        parser.error('--models accepts YOLOv8/11/26 checkpoint basenames only')
    root.mkdir(parents=True, exist_ok=True)
    report = dict(platform=platform.platform(), processor=platform.processor(),
                  scope='inference incl. preprocessing/NMS, excludes decode and first-run export',
                  quality='unlabeled frames: predictions measure agreement, not accuracy',
                  frames=[p.name for p in sorted(frames.glob('*.png'))], results=[])
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report['gpu'] = None
    matrix = [('torch', 'cpu', 1), ('torch', 'cpu', 8)] if 'torch' in args.runtimes else []
    matrix += [(runtime, 'cpu', 1) for runtime in args.runtimes if runtime != 'torch']
    if args.gpu:
        matrix.append(('torch', '0', 8))
    script = str(Path(__file__).resolve())
    for name in args.models:
        for runtime, device, batch in matrix:
            key = f'{Path(name).stem}-{runtime}-{device}-{batch}'
            result_path, log_path = root / f'{key}.json', root / f'{key}.log'
            settings = dict(workspace=str(root), frames=str(frames), model=name,
                            runtime=runtime, device=device, batch=batch, imgsz=args.imgsz,
                            repeats=args.repeats, result=str(result_path))
            settings_path = root / f'{key}-settings.json'
            settings_path.write_text(json.dumps(settings))
            print(f'Benchmarking {key} ...', flush=True)
            with log_path.open('w') as log:
                try:
                    process = subprocess.run([sys.executable, script, '--worker-settings', str(settings_path)],
                                             cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    if process.returncode:
                        raise RuntimeError(f'Worker exited {process.returncode}; see {log_path.name}')
                    result = json.loads(result_path.read_text())
                    print(f"  {result['ms_per_frame']:.1f} ms/frame; "
                          f"{result['inference_realtime_at_5fps']:.2f}x inference-only", flush=True)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    result = dict(model=name, runtime=runtime, device=device, batch=batch, error=str(error))
                    print(f'  FAILED: {error}', flush=True)
            report['results'].append(result)
            # Incremental report survives interruption. The output workspace was
            # required empty, so no existing user artifacts can be overwritten.
            temporary = root / 'report.pending.json'
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(root / 'report.json')
    print(f'Report: {root / "report.json"}', flush=True)
    return int(any('error' in row for row in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
