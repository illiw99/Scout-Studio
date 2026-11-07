"""Lazy video/frame encoders with one RGB uint8 batch input and numeric output.

Backends consume supplied local assets only; inference never downloads weights.
"""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import numpy as np

ENCODERS = {
    'mobilenet-v3-small': dict(size=96, frames=1, width=576, label='MobileNet V3 Small + temporal head'),
    's3d': dict(size=112, frames=16, width=400, label='S3D video network + temporal head'),
}


def torch_encoder(kind, weights):
    import torch
    from torchvision.models import mobilenet_v3_small
    from torchvision.models.video import s3d
    if kind not in ENCODERS:
        raise ValueError('Unknown visual encoder')
    network = mobilenet_v3_small(weights=None) if kind == 'mobilenet-v3-small' else s3d(weights=None)
    network.load_state_dict(torch.load(weights if hasattr(weights, 'read') else str(weights), map_location='cpu', weights_only=True))
    body = (torch.nn.Sequential(network.features, network.avgpool, torch.nn.Flatten(1))
            if kind == 'mobilenet-v3-small' else
            torch.nn.Sequential(network.features, torch.nn.AdaptiveAvgPool3d(1), network.classifier, torch.nn.Flatten(1)))
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.body = body
            mean, std = (([.485,.456,.406],[.229,.224,.225]) if kind == 'mobilenet-v3-small'
                         else ([.43216,.394666,.37645],[.22803,.22145,.216989]))
            shape = (1,3,1,1) if kind == 'mobilenet-v3-small' else (1,3,1,1,1)
            self.register_buffer('mean', torch.tensor(mean).reshape(shape))
            self.register_buffer('std', torch.tensor(std).reshape(shape))
        def forward(self, rgb):
            rgb = rgb.permute(0,3,1,2) if kind == 'mobilenet-v3-small' else rgb.permute(0,4,1,2,3)
            return self.body((rgb.float()/255-self.mean)/self.std)
    return Encoder().eval()


class VisualEncoder:
    def __init__(self, kind, path, *, device='cpu', sha256=None, threads=4):
        if kind not in ENCODERS or not isinstance(device,str) or (device!='cpu' and not device.startswith('cuda')):
            raise ValueError('Invalid visual encoder configuration')
        self.kind, self.spec, self.device = kind, ENCODERS[kind], device
        path = Path(path)
        payload=path.read_bytes()
        if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('Visual encoder checksum mismatch')
        self.session = None
        if path.suffix == '.onnx':
            if device != 'cpu': raise ValueError('ONNX visual encoders currently use CPU')
            import onnxruntime as ort
            options = ort.SessionOptions(); options.intra_op_num_threads = threads
            self.session = ort.InferenceSession(payload, sess_options=options, providers=['CPUExecutionProvider'])
        else:
            import torch
            torch.set_num_threads(threads)
            self.network = torch_encoder(kind, io.BytesIO(payload)).to(device)
    def predict(self, rgb):
        rgb = np.asarray(rgb)
        spec = self.spec
        shape = (spec['size'],spec['size'],3) if spec['frames'] == 1 else (spec['frames'],spec['size'],spec['size'],3)
        if rgb.dtype != np.uint8 or rgb.ndim != len(shape)+1 or tuple(rgb.shape[1:]) != shape:
            raise ValueError('Encoder expects correctly sized RGB uint8 batches')
        if not len(rgb): return np.empty((0,spec['width']), np.float32)
        if self.session:
            result = self.session.run(None, {self.session.get_inputs()[0].name: rgb})[0]
        else:
            import torch
            with torch.inference_mode():
                result = self.network(torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)).cpu().numpy()
        result = np.asarray(result, np.float32)
        if result.shape != (len(rgb),spec['width']) or not np.isfinite(result).all():
            raise ValueError('Invalid visual encoder output')
        return result


def encode_source_many(path, start, encoders, *, device='cpu', batch=32, rates=None, progress=None):
    """Bounded streaming windows; at most one batch and 16 frames per encoder.

    Centered S3D context repeats source-edge frames exactly as the original
    offline extractor. Output features, not decoded video, scale with duration.
    """
    import time
    import cv2
    from collections import deque
    from contextlib import closing
    from . import media
    if batch<1:raise ValueError('Encoder batch must be positive')
    before=Path(path).stat();began=time.perf_counter();decode_seconds=0.;states={}
    for key,encoder in encoders.items():
        rate=(rates or {}).get(key,2.5 if encoder.spec['frames']==1 else 1.)
        step=int(round(5/rate))
        if step<1 or abs(5/step-rate)>1e-8:raise ValueError('Visual sampling must divide the 5Hz decode clock')
        states[key]=dict(encoder=encoder,frames=deque(maxlen=encoder.spec['frames']),next=0,step=step,
                         pending=[],centers=[],parts=[],seconds=0.,first=None)
    def flush(key,state):
        if not state['pending']:return
        clock=time.perf_counter()
        state['parts'].append(state['encoder'].predict(np.stack(state['pending'])))
        state['seconds']+=time.perf_counter()-clock
        state['pending'].clear()
        if progress:progress(key,len(state['centers']))
    def append(key,state,center,last):
        count=state['encoder'].spec['frames'];window=state['frames']
        if count==1:rgb=window[-1]
        else:
            indices=np.clip(np.arange(center-7,center+9),0,last);oldest=last-len(window)+1
            rgb=np.stack([state['first'] if i==0 else window[int(i-oldest)] for i in indices])
        state['pending'].append(rgb);state['centers'].append(center)
        if len(state['pending'])>=batch:flush(key,state)
    count=0
    reader=media.iter_frames(path,5,320,180,hwaccel=device!='cpu',gpu_resize=device!='cpu')
    with closing(reader):
        while True:
            clock=time.perf_counter()
            try:frame=next(reader)
            except StopIteration:
                decode_seconds+=time.perf_counter()-clock;break
            decode_seconds+=time.perf_counter()-clock
            for key,state in states.items():
                encoder=state['encoder'];n=encoder.spec['frames']
                if n==1 and count!=state['next']:continue
                clock=time.perf_counter();size=encoder.spec['size']
                small=cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA)
                if state['first'] is None:state['first']=small
                state['frames'].append(small)
                state['seconds']+=time.perf_counter()-clock
                future=8 if n==16 else 0
                if count>=state['next']+future:
                    append(key,state,state['next'],count);state['next']+=state['step']
            count+=1
    if not count:raise ValueError('No decoded frames')
    result={}
    for key,state in states.items():
        while state['next']<count:
            append(key,state,state['next'],count-1);state['next']+=state['step']
        flush(key,state)
        result[key]=(start+np.asarray(state['centers'])/5,np.concatenate(state['parts']),
                     dict(decode_seconds=decode_seconds,encoder_seconds=state['seconds'],source_wall_seconds=time.perf_counter()-began))
    after=Path(path).stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Video changed while encoding')
    return result


def encode_source(path,start,encoder,*,device='cpu',batch=32,sample_fps=None,progress=None):
    rates=None if sample_fps is None else {encoder.kind:sample_fps}
    return encode_source_many(path,start,{encoder.kind:encoder},device=device,batch=batch,rates=rates,progress=progress)[encoder.kind]
