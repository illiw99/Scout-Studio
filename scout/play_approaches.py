"""Exchangeable play scorers: named numeric observations -> timestamp probabilities.

The detector contract concerns boxes. This separate contract also accommodates
whole-scene video encoders. Optional runtimes are imported only when selected.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Observations:
    t: np.ndarray
    X: np.ndarray
    names: tuple[str, ...]

    def validate(self):
        if (self.t.ndim != 1 or self.X.shape != (len(self.t),len(self.names))
                or not len(self.t) or not np.isfinite(self.t).all() or np.any(np.diff(self.t)<=0)
                or np.any(self.t<0) or np.isinf(self.X).any() or len(set(self.names)) != len(self.names)):
            raise ValueError('Invalid named observation sequence')


@dataclass(frozen=True)
class PlayScores:
    t: np.ndarray
    probabilities: np.ndarray

    def validate(self):
        if (self.t.ndim != 1 or self.probabilities.shape != self.t.shape
                or not np.isfinite(self.t).all() or np.any(np.diff(self.t)<=0)
                or not np.isfinite(self.probabilities).all()
                or np.any((self.probabilities<0)|(self.probabilities>1))):
            raise ValueError('Invalid timestamp probabilities')


class PlayPredictor(Protocol):
    def predict(self, observations: Observations) -> PlayScores: ...


@dataclass(frozen=True)
class ApproachInfo:
    id: str
    label: str
    input_kind: str


POSTPROCESS_DEFAULTS = dict(model_threshold=.5, model_low_ratio=.45, model_smooth=.6,
    model_min_duration=2., merge_gap=1.5, model_peak_threshold=0.,
    model_refine_fraction=0., model_refine_padding=0., model_pre_roll=0., model_post_roll=0.)


def complete_postprocessing(parameters):
    if set(parameters)-set(POSTPROCESS_DEFAULTS):raise ValueError('Unknown interval setting')
    return dict(POSTPROCESS_DEFAULTS,**parameters)


def prepare_features(X, transform):
    if transform=='raw':return X
    if transform!='game-zscore':raise ValueError('Unknown pipeline feature transform')
    X=np.asarray(X,np.float32)
    count=np.isfinite(X).sum(axis=0)
    mean=np.nansum(X,axis=0)/np.maximum(count,1)
    scale=np.sqrt(np.nansum((X-mean)**2,axis=0)/np.maximum(count,1))
    return ((X-mean)/np.maximum(scale,.05)).astype(np.float32)


_registry = {}


def register_approach(info, factory, *, provider=None):
    if (not isinstance(info,ApproachInfo) or not isinstance(info.id,str) or not info.id
            or not isinstance(info.input_kind,str) or not info.input_kind or info.id in _registry
            or not callable(factory) or (provider is not None and not callable(provider))):
        raise ValueError('Approach requires unique metadata and an explicit factory')
    _registry[info.id] = (info,factory,provider)


def list_approaches():
    return tuple(value[0] for value in _registry.values())


def sequence_windows(t, X, offsets):
    """Nearest observed context; clamp at game edges, never cross games."""
    if (not len(t) or t.ndim!=1 or len(X)!=len(t) or np.any(np.diff(t)<=0)
            or not offsets or 0 not in offsets or any(not np.isfinite(o) for o in offsets)):
        raise ValueError('Context must include zero')
    columns=[]
    for offset in offsets:
        wanted=t+offset;right=np.clip(np.searchsorted(t,wanted),0,len(t)-1);left=np.maximum(0,right-1)
        chosen=np.where(np.abs(t[left]-wanted)<=np.abs(t[right]-wanted),left,right)
        columns.append(chosen)
    return np.stack(columns,axis=1)


def visual_subsample(t,X,clips,base_fps,target_fps):
    factor=round(base_fps/target_fps)
    if factor<1 or abs(base_fps/factor-target_fps)>1e-8:raise ValueError('Invalid visual subsampling rate')
    starts=np.asarray([source['start'] for source in clips],float)
    if not len(starts) or np.any(np.diff(starts)<=0):raise ValueError('Source starts must be ordered')
    # Source-duration rounding can overlap adjacent source intervals. Assign
    # every observation once by its source start instead of overlapping masks.
    owner=np.clip(np.searchsorted(starts,t,side='right')-1,0,len(starts)-1)
    clocks=np.rint((t-starts[owner])*base_fps).astype(np.int64)
    rows=np.flatnonzero(clocks%factor==0)
    return t[rows],X[rows]


def read_manifest(path, *, verify=False, expected_sha256=None):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>1024*1024:
        raise ValueError('Invalid play pipeline manifest')
    raw=path.read_bytes()
    if len(raw)>1024*1024 or (expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256):
        raise ValueError('Play pipeline manifest identity changed')
    spec=json.loads(raw)
    if (not isinstance(spec,dict) or spec.get('format')!='scout-play-pipeline-v1'
            or not isinstance(spec.get('approach'),str) or spec['approach'] not in _registry):
        raise ValueError('Unsupported play pipeline')
    if _registry[spec['approach']][0].input_kind=='person-features' and (
            not isinstance(spec.get('requires_model_pack'),str) or '@' not in spec['requires_model_pack']):
        raise ValueError('Person pipelines must bind an exact detector model pack')
    if not isinstance(spec.get('id'),str) or not re.fullmatch('[a-z0-9][a-z0-9._-]{0,95}',spec['id']):
        raise ValueError('Invalid play pipeline identity')
    if (not isinstance(spec.get('label',spec['id']),str) or len(spec.get('label',''))>160
            or not isinstance(spec.get('evaluation_note',''),str)):
        raise ValueError('Invalid pipeline display metadata')
    names=spec.get('names');offsets=spec.get('offsets');fps=spec.get('sample_fps')
    if (not isinstance(names,list) or not names or not all(isinstance(n,str) for n in names)
            or len(set(names))!=len(names) or len(names)>2048 or type(fps) not in (int,float)
            or not np.isfinite(fps) or fps<.01 or fps>30 or not isinstance(offsets,list)
            or not 1<=len(offsets)<=101 or 0 not in offsets
            or any(type(o) not in (int,float) or not np.isfinite(o) or abs(o)>60 for o in offsets)
            or any(b<=a for a,b in zip(offsets,offsets[1:]))):
        raise ValueError('Invalid play feature schema/context')
    assets=spec.get('assets')
    if not isinstance(assets,dict) or 'head' not in assets:raise ValueError('Missing pipeline assets')
    for asset in assets.values():
        if not isinstance(asset,dict):raise ValueError('Invalid pipeline asset')
        name=asset.get('file');digest=asset.get('sha256')
        if (not isinstance(name,str) or not name or Path(name).name!=name or '\\' in name or name in ('.','..')
                or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)):
            raise ValueError('Invalid pipeline asset identity')
        source=path.parent/name
        if source.is_symlink() or not source.is_file() or source.stat().st_size>128*1024*1024:
            raise ValueError('Missing or oversized pipeline asset')
        if verify and hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Pipeline asset checksum mismatch')
    post=spec.get('postprocessing',{})
    allowed=set(POSTPROCESS_DEFAULTS)
    if not isinstance(post,dict) or set(post)!=allowed or any(type(x) not in (int,float) or not np.isfinite(x) or x<0 for x in post.values()):
        raise ValueError('Invalid play postprocessing')
    if any(post[k]>1 for k in ('model_threshold','model_low_ratio','model_peak_threshold','model_refine_fraction')):
        raise ValueError('Probability interval settings must be within zero and one')
    if spec.get('feature_transform','raw') not in ('raw','game-zscore') or spec.get('feature_dtype','float32') not in ('float16','float32'):
        raise ValueError('Unknown feature preprocessing')
    kind=_registry[spec['approach']][0].input_kind
    if kind in ('s3d','mobilenet-v3-small'):
        from .video_encoders import ENCODERS
        if (type(spec.get('encoder_revision')) is not int or spec['encoder_revision']!=1 or len(names)!=ENCODERS[kind]['width']
                or 'encoder_cpu' not in assets or not assets['encoder_cpu']['file'].endswith('.onnx')
                or fps>5 or abs(5/round(5/fps)-fps)>1e-8):
            raise ValueError('Visual pipeline encoder/schema does not match revision 1')
        if 'encoder_reference' in assets and Path(assets['encoder_reference']['file']).suffix not in ('.pt','.pth'):
            raise ValueError('Reference visual encoders require verified tensor weights')
    return spec


class OnnxPlayPredictor:
    def __init__(self, path, *, threads=4, expected_manifest_sha256=None):
        import onnxruntime as ort
        self.path=Path(path);self.spec=read_manifest(path,verify=True,expected_sha256=expected_manifest_sha256)
        options=ort.SessionOptions();options.intra_op_num_threads=threads
        payload=(self.path.parent/self.spec['assets']['head']['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=self.spec['assets']['head']['sha256']:raise ValueError('Head changed while loading')
        self.session=ort.InferenceSession(payload,sess_options=options,providers=['CPUExecutionProvider'])
    def predict(self, observations):
        observations.validate()
        if tuple(self.spec['names'])!=tuple(observations.names):raise ValueError('Play feature names/order differ')
        X=observations.X
        if self.spec.get('feature_dtype')=='float16':X=X.astype(np.float16).astype(np.float32)
        X=prepare_features(X,self.spec.get('feature_transform','raw'))
        indices=sequence_windows(observations.t,X,self.spec['offsets']);parts=[]
        for first in range(0,len(indices),256):
            batch=X[indices[first:first+256]].astype(np.float32)
            parts.append(self.session.run(None,{self.session.get_inputs()[0].name:batch})[0].reshape(-1))
        result=PlayScores(observations.t.copy(),np.concatenate(parts).astype(np.float64));result.validate();return result


def create_predictor(path, **kwargs):
    spec=read_manifest(path,expected_sha256=kwargs.get('expected_manifest_sha256'))
    return _registry[spec['approach']][1](path,**kwargs)


def available_pipelines(directory):
    directory=Path(directory)
    result=[]
    for path in sorted(directory.glob('*/pipeline.json')):
        try:spec=read_manifest(path)
        except (OSError,ValueError,TypeError,KeyError):continue
        result.append((path,spec))
    return result


for info in (
    ApproachInfo('person-temporal','Person features + temporal model','person-features'),
    ApproachInfo('s3d-video','S3D video network + temporal model','s3d'),
    ApproachInfo('mobile-temporal','MobileNet V3 Small + temporal model','mobilenet-v3-small'),
):
    register_approach(info,OnnxPlayPredictor)


def pipeline_info(spec):
    return _registry[spec['approach']][0]


def run_visual_pipeline(cfg, clips, path):
    """Run source-local streaming encoders, with all-or-nothing auto CPU retry."""
    from .video_encoders import VisualEncoder,encode_source
    import time
    started=time.perf_counter();predictor=create_predictor(path,threads=max(1,cfg.inference_threads or 4),expected_manifest_sha256=getattr(cfg,'play_pipeline_sha256',None))
    spec=predictor.spec;kind=pipeline_info(spec).input_kind
    if kind=='person-features':raise ValueError('Person pipeline requires person observations')
    automatic=cfg.device=='auto';requested='cpu' if cfg.device=='cpu' else 'cuda'
    reason=None
    if requested!='cpu':
        try:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('No available CUDA device')
        except (ImportError,RuntimeError) as exc:
            if not automatic:raise RuntimeError('Requested GPU pipeline is unavailable') from exc
            requested='cpu';reason=str(exc)
    def attempt(device):
        key='encoder_reference' if device!='cpu' else 'encoder_cpu'
        if key not in spec['assets']:raise RuntimeError('Pipeline lacks the requested encoder runtime')
        asset=spec['assets'][key]
        target=device
        if device!='cpu' and not automatic:
            target=str(cfg.device)
            if target.isdigit():target='cuda:'+target
        encoder=VisualEncoder(kind,Path(path).parent/asset['file'],device=target,sha256=asset['sha256'],threads=max(1,cfg.inference_threads or 4))
        times=[];values=[];rows=[]
        for clip in clips:
            def progress(_kind,count):
                percent=min(100.,100*count/max(1,clip.info.duration*spec['sample_fps']))
                print(f'  {clip.path.name}: {percent:.1f}% {kind}',flush=True)
            t,X,timing=encode_source(clip.path,clip.start,encoder,device=target,sample_fps=spec['sample_fps'],progress=progress)
            times.append(t);values.append(X);rows.append(dict(source=clip.path.name,**timing))
        X=np.concatenate(values)
        if spec.get('feature_dtype')=='float16':X=X.astype(np.float16).astype(np.float32)
        scores=predictor.predict(Observations(np.concatenate(times),X,tuple(spec['names'])))
        return scores,dict(device=target,seconds=time.perf_counter()-started,sources=rows)
    if requested!='cpu':
        try:return attempt('cuda')
        except (RuntimeError,ImportError) as exc:
            if not automatic:raise
            reason=str(exc)
            print('  visual pipeline: GPU execution failed; retrying on CPU',flush=True)
    scores,timing=attempt('cpu')
    if reason:timing['fallback_reason']=reason
    return scores,timing


@dataclass(frozen=True)
class PipelineResult:
    plays: list
    scores: PlayScores | None
    timings: dict

    @property
    def intervals(self):
        return np.asarray([(p.start,p.end) for p in self.plays],np.float64).reshape(-1,2)


class PlayDetectionPipeline(Protocol):
    def predict(self, clips, cfg) -> PipelineResult: ...


class BaselinePipeline:
    """Adapter for the established detector/features/HGB-or-rules pipeline."""
    def predict(self, clips, cfg):
        import time
        from .cli import _detect_for,_features_for,_segment
        started=time.perf_counter();_detect_for(cfg,clips);observations=_features_for(cfg,clips)
        plays=_segment(cfg,observations,clips[-1].end)
        return PipelineResult(plays,None,dict(seconds=time.perf_counter()-started,input_kind='person-features'))


class ArtifactPipeline:
    """A complete bound preprocessing + scorer + interval-generation pipeline."""
    def __init__(self,path):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.spec=read_manifest(path,expected_sha256=self.sha256)
    def predict(self, clips, cfg):
        import time
        from . import model,model_features
        from .cli import _detect_for,_features_for
        started=time.perf_counter();spec=read_manifest(self.path,verify=True,expected_sha256=self.sha256)
        expected=getattr(cfg,'play_pipeline_sha256',None)
        if expected and hashlib.sha256(self.path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Selected play pipeline changed after it was queued')
        cfg.play_pipeline_sha256=self.sha256
        kind=pipeline_info(spec).input_kind
        if kind=='person-features':
            from .model_packs import ModelStore
            cfg.detector_adapter=None
            ModelStore(cfg.model_packs_dir).apply(cfg,spec['requires_model_pack'])
            if cfg.sample_fps!=spec['sample_fps']:raise ValueError('Scorer sampling differs from its detector pack')
        else:
            cfg.sample_fps=spec['sample_fps']
        provider=_registry[spec['approach']][2]
        if provider is not None:
            observations,timing=provider(clips,cfg,spec,self.path)
            scores=create_predictor(self.path,expected_manifest_sha256=self.sha256).predict(observations)
        elif kind=='person-features':
            _detect_for(cfg,clips);f=_features_for(cfg,clips)
            X,names=model_features.expand(f.X,f.names,cfg.sample_fps,cfg.model_feature_set)
            if not set(spec['names'])<=set(names):raise ValueError('Person features incompatible with selected pipeline')
            X=X[:,[names.index(n) for n in spec['names']]]
            predictor=create_predictor(self.path,threads=max(1,cfg.inference_threads or 4),expected_manifest_sha256=self.sha256)
            scores=predictor.predict(Observations(f.t,X,tuple(spec['names'])))
            timing=dict(input_kind=kind)
        else:
            scores,timing=run_visual_pipeline(cfg,clips,self.path)
        scores.validate()
        if (len(scores.t) and (scores.t[0]<getattr(clips[0],'start',0.)-.002 or scores.t[-1]>clips[-1].end+.002)):
            raise ValueError('Pipeline observation timestamps fall outside source timeline')
        for key,value in spec['postprocessing'].items():setattr(cfg,key,value)
        plays=model.to_plays(cfg,model.postprocess(cfg,scores.t,scores.probabilities),clips[-1].end)
        timing['seconds']=time.perf_counter()-started
        return PipelineResult(plays,scores,timing)


def create_pipeline(path=None):
    return BaselinePipeline() if path is None else ArtifactPipeline(path)


def install_pipeline(path,store,*,probe=True):
    """Install verified immutable assets only; never copy training/GT directories."""
    import os,shutil,tempfile
    path=Path(path);raw=path.read_bytes();identity=hashlib.sha256(raw).hexdigest()
    spec=read_manifest(path,verify=True,expected_sha256=identity)
    # Probe the portable head off the UI thread before publishing any selection.
    if probe:
        predictor=create_predictor(path,expected_manifest_sha256=identity)
        observations=Observations(np.array([0.,1.]),np.zeros((2,len(spec['names'])),np.float32),tuple(spec['names']))
        predictor.predict(observations)
    store=Path(store);store.mkdir(parents=True,exist_ok=True)
    if path.read_bytes()!=raw:raise ValueError('Pipeline changed during installation')
    target=store/(spec['id']+'-'+identity[:12])
    if target.exists():
        if (target/'pipeline.json').read_bytes()!=raw:raise ValueError('Pipeline installation identity conflict')
        read_manifest(target/'pipeline.json',verify=True);return target/'pipeline.json'
    with tempfile.TemporaryDirectory(prefix='.install-',dir=store) as directory:
        stage=Path(directory)/'pipeline';stage.mkdir()
        (stage/'pipeline.json').write_bytes(raw)
        for asset in spec['assets'].values():
            source=path.parent/asset['file'];destination=stage/asset['file']
            if not destination.exists():shutil.copyfile(source,destination)
        read_manifest(stage/'pipeline.json',verify=True)
        os.rename(stage,target)
    return target/'pipeline.json'


def default_pipeline_store():
    from .model_packs import default_store
    return default_store().parent/'play-pipelines'
