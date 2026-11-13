"""Headless frozen processing entry point. Never import the desktop package."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import importlib.util


def load_engine_identity():
    identity = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'engine-id.json'
    if identity.is_file():
        value = json.loads(identity.read_text(encoding='utf-8'))
        os.environ['SCOUT_ENGINE_ID'] = value['engine_id']
        sys._scout_engine_identity = value


load_engine_identity()


def engine_probe(destination):
    """Exercise a real CPU operator even when this is a CUDA wheel without a GPU."""
    from scout import __version__
    report = {'application_version': __version__, 'schema': 1, 'ok': False, 'engine_id': os.environ.get('SCOUT_ENGINE_ID', ''),
              'devices': ['cpu'], 'versions': {}, 'nms': False}
    with tempfile.TemporaryDirectory(prefix='scout-engine-probe-') as temporary:
        root = Path(temporary)
        settings = dict(YOLO_CONFIG_DIR=str(root / 'yolo'), MPLCONFIGDIR=str(root / 'matplotlib'),
                        MPLBACKEND='Agg', ULTRALYTICS_OFFLINE='true', YOLO_AUTOINSTALL='false')
        previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)
        try:
            import torch
            import torchvision
            import ultralytics
            from sklearn.ensemble import RandomForestClassifier
            import matplotlib.pyplot as plt
            boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device='cpu')
            kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8]), .5)
            if kept.tolist() != [0]:
                raise RuntimeError('TorchVision CPU NMS returned an invalid result')
            report['nms'] = True
            RandomForestClassifier(n_estimators=1, random_state=1).fit([[0], [1]], [0, 1]).predict([[0]])
            figure = plt.figure()
            figure.savefig(root / 'plot.png')
            plt.close(figure)
            report['versions'] = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                      ultralytics=ultralytics.__version__)
            report['cuda_build'] = torch.version.cuda
            if torch.cuda.is_available():
                try:
                    gpu_boxes = boxes.to('cuda')
                    if torchvision.ops.nms(gpu_boxes, torch.tensor([.9, .8], device='cuda'), .5).tolist() != [0]:
                        raise RuntimeError('CUDA NMS returned an invalid result')
                    report['devices'].append('cuda')
                except Exception as exc:
                    report['gpu_error'] = str(exc)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    if torch.ones(2, device='mps').sum().item() != 2:
                        raise RuntimeError('MPS tensor operation failed')
                    report['devices'].append('mps')
                except Exception as exc:
                    report['gpu_error'] = str(exc)
            probe_inference(root, report)
            from scout.model_catalog import directory, install_pipelines
            if directory() is None:
                raise RuntimeError('Processing engine lacks its bundled decision catalogue')
            install_pipelines(root / 'pipeline-probe')
            report['bundled_pipelines'] = True
            report['ok'] = True
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    # The frozen identity is build data, never inferred from the caller's machine.
    identity = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'engine-id.json'
    if identity.is_file():
        value = json.loads(identity.read_text(encoding='utf-8'))
        report['engine_id'] = value['engine_id']
        for name in ('requirements_sha256', 'wheel_sha256'):
            if name in value:
                report[name] = value[name]
    Path(destination).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['ok'] else 1


def probe_inference(root, report):
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from scout.config import Config
    from scout.inference import CpuDetector
    raw = np.zeros((1, 84, 1), dtype=np.float32)
    raw[0, :4, 0] = [50, 50, 20, 20]
    raw[0, 4, 0] = .9
    graph = helper.make_model(helper.make_graph([
        helper.make_node('Constant', [], ['output0'], value=numpy_helper.from_array(raw))],
        'engine-cpu-fixture',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 'height', 'width'])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 84, 1])]),
        opset_imports=[helper.make_opsetid('', 17)])
    graph.ir_version = 10
    graph_file = root / 'fixture.onnx'
    onnx.save(graph, str(graph_file))
    frame = np.zeros((96, 160, 3), dtype=np.uint8)
    prediction = CpuDetector(Config(model=str(graph_file), detector_backend='onnx', device='cpu', imgsz=160)).predict([frame])[0]
    if not np.allclose(prediction.boxes, [[40, 40, 60, 60]]):
        raise RuntimeError('ONNX CPU detector fixture mismatch')
    report['inference'] = True
    if importlib.util.find_spec('openvino') is not None:
        import openvino as ov
        xml = root / 'fixture.xml'
        ov.save_model(ov.Core().read_model(str(graph_file)), str(xml), compress_to_fp16=False)
        exported = CpuDetector(Config(model=str(xml), detector_backend='openvino', device='cpu', imgsz=160)).predict([frame])[0]
        if not np.allclose(exported.boxes, prediction.boxes):
            raise RuntimeError('OpenVINO CPU detector fixture mismatch')
        report['openvino'] = True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == '--engine-probe':
        return engine_probe(argv[1])
    if argv and argv[0] == '--worker-job':
        from scout.worker import main as worker
        return worker(argv[1:])
    if argv and argv[0] == '--scout-cli':
        from scout.cli import main as cli
        return cli(argv[1:])
    raise SystemExit('Usage: Scout-Worker --worker-job DESCRIPTOR | --scout-cli ARGS... | --engine-probe REPORT.json')


if __name__ == '__main__':
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    multiprocessing.freeze_support()
    raise SystemExit(main())
