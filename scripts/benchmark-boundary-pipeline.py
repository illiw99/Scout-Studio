#!/usr/bin/env python3
"""Bounded real-media pipeline timing; all generated state stays in a new workspace.

Includes decode, detection, field adaptation, tracking, features, HGB and refiner.
Excludes one-time downloads/exports and video export. Reads original fields only.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--duration',type=float,default=120)
    parser.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.start<0 or args.duration<=0 or args.repeats<1 or args.threads<1:parser.error('Invalid bounds')
    root=args.workspace.resolve();root.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_CONFIG_DIR']=str(root/'ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib')
    os.environ['YOLO_AUTOINSTALL']='false'
    import numpy as np
    from threadpoolctl import threadpool_limits
    from scout import __version__,detect,media,model,pick,inference
    from scout.cli import _detect_for,_features_for,_region_for
    from scout.config import Config
    from scout.model_defaults import ensure_default
    from scout.model_catalog import directory,prepare_default_bundle
    from scout.model_packs import ModelStore
    from scout.timeline import Clip
    source=args.source.resolve();before=source.stat();info=media.probe(source)
    if args.start+args.duration>info.duration:raise ValueError('Window exceeds source')
    clip=Clip(replace(info,duration=args.duration),0,0)
    cfg=Config(source_dir=str(source.parent),device=args.device,inference_threads=args.threads,
        inference_cache_dir=str(root/'detectors'),cache_dir=str(root/'cache-initial'),out_dir=str(root/'output'))
    field=pick.field_path(cfg,clip)
    fields_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in field.parent.glob('*.json')}
    if _region_for(cfg,[clip],clip) is None:raise ValueError('Benchmark requires a saved field')
    store=ModelStore(root/'packs');bundle=prepare_default_bundle(directory(),root/'bundle');ensure_default(store,bundle)
    selection=store.active();spec=json.loads((store.folder(selection)/'manifest.json').read_text())
    ref=spec['gpu_reference']
    cached=inference.reference_path(Config(),ref['name'],ref['sha256'])
    target=inference.reference_path(cfg,ref['name'],ref['sha256'])
    if cached.is_file() and inference.digest(cached)==ref['sha256']:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cached,target)
    started=time.perf_counter();store.apply(cfg,selection);classifier,_=model.load(Path(cfg.model_path),cfg)
    preparation=time.perf_counter()-started
    if args.device=='cpu' and (cfg.device!='cpu' or cfg.decode_gpu):raise ValueError('CPU benchmark selected acceleration')
    decoder=detect.iter_frames
    def bounded(path,fps,width,height,**kwargs):
        if Path(path).resolve()!=source:raise ValueError('Unexpected benchmark decode source')
        return decoder(path,fps,width,height,start=args.start,duration=args.duration,**kwargs)
    rows=[]
    with threadpool_limits(limits=args.threads):
        try:
            detect.iter_frames=bounded
            for repeat in range(args.repeats):
                cfg.cache_dir=str(root/f'cache-{repeat}')
                begun=time.perf_counter();_detect_for(cfg,[clip]);detected=time.perf_counter()
                observations=_features_for(cfg,[clip]);featured=time.perf_counter()
                X=model.prediction_inputs(observations,cfg);probability=classifier.predict_proba(X)[:,1]
                spans=model.postprocess(cfg,observations.t,probability)
                spans=[(max(0,a),min(args.duration,b)) for a,b in spans if min(args.duration,b)>max(0,a)]
                scored=time.perf_counter()
                refined=classifier.boundary_refiner.refine(observations.t,X,probability,spans,args.duration)
                plays=model.to_plays(cfg,refined,args.duration);finished=time.perf_counter()
                row=dict(repeat=repeat,video_seconds=args.duration,frames=len(observations.t),plays=len(plays),
                    decode_detection_seconds=detected-begun,field_tracking_features_seconds=featured-detected,
                    decision_seconds=scored-featured,refiner_seconds=finished-scored,
                    total_seconds=finished-begun,realtime=args.duration/(finished-begun))
                rows.append(row);print(json.dumps(row),flush=True)
        finally:detect.iter_frames=decoder
    after=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert fields_before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in field.parent.glob('*.json')}
    report=dict(version=__version__,selection=selection,device=cfg.device,backend=cfg.detector_backend,
        imgsz=cfg.imgsz,fps=cfg.sample_fps,threads=cfg.inference_threads,batch=cfg.batch,decode_gpu=cfg.decode_gpu,
        preparation_seconds=preparation,source=str(source),source_start=args.start,source_unchanged=True,fields_unchanged=True,
        scope=__doc__,rows=rows,median_realtime=args.duration/float(np.median([r['total_seconds'] for r in rows])),
        versions={name:inference.version(name) for name in ('torch','ultralytics','openvino','numpy')})
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(median_realtime=report['median_realtime'],report=str(root/'report.json'))))


if __name__=='__main__':main()
