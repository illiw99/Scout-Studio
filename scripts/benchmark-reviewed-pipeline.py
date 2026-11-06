#!/usr/bin/env python3
"""Time an isolated reviewed-game candidate, including decoding and play prediction.

Reads only frozen development inputs. Fits its own head outside the timing;
does not measure generalization. CPU mode disables both CUDA inference and decode.
"""
import argparse
from contextlib import closing
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import detect,features,groundtruth,inference,media,model
from scout.config import Config
from scout.cli import _region_for
from scout.groundtruth import Marked
from scout.model_evaluation import project_field_region
from scout.storage import validate_name


def helper(filename):
    spec=importlib.util.spec_from_file_location(filename.replace('-','_'),Path(__file__).with_name(filename+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class SourceGeometryDetector:
    def __init__(self,engine,geometry):self.engine=engine;self.geometry=geometry
    def validate_source(self,cfg):
        if hasattr(self.engine,'validate_source'):self.engine.validate_source(cfg)
    def predict(self,frames,**kwargs):
        results=self.engine.predict(frames,**kwargs);mapped=[]
        if len(results)!=len(frames):raise ValueError('Unexpected detector batch result count')
        for frame,result in zip(frames,results):
            if isinstance(result,inference.Prediction):boxes,conf=result.boxes,result.conf
            else:boxes,conf=result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()
            h,w=frame.shape[:2];ow,oh=self.geometry
            mapped.append(inference.Prediction(np.asarray(boxes)*np.array([ow/w,oh/h,ow/w,oh/h]),conf))
        return mapped


def active_window_start(marks,start,source_duration,duration):
    first=next((a for a,b in marks if a>=start and b<=start+source_duration),None)
    desired=(source_duration-duration)/2 if first is None else first-start-30
    return max(0.,min(source_duration-duration,desired))


def run_source(cfg,source,clips,record,identity,args,engine,regions,scaled,specification,
               duration,offset=0.,classifier=None,best=None,post=None):
    """Private cache-backed source stage shared by timing and frozen confirmation."""
    before=source.path.stat()
    if source.path.name!=identity['source'] or (before.st_size,before.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):
        raise ValueError('Source identity changed')
    clip=replace(source,start=source.start+offset,info=replace(source.info,duration=duration))
    shape=scaled.resized_shape(clip.info.width,clip.info.height,args.size);captured={}
    indices={int(round(f*(round(duration*cfg.sample_fps)-1))) for f in (0,.15,.325,.5,.675,.85,1)}
    def decode(path,fps,width,height,*,cancel_event=None):
        frames=(scaled.cuda_frames(path,fps,*shape,duration=duration,start=offset) if args.backend=='cuda' else
            media.iter_frames(path,fps,*shape,start=offset,duration=duration,hwaccel=False,cancel_event=cancel_event))
        with closing(frames):
            for i,rgb in enumerate(frames):
                if cancel_event is not None and cancel_event.is_set():return
                if i in indices:captured[i]=rgb.copy()
                yield rgb
    original_decoder=detect.iter_frames
    detect.iter_frames=decode
    started=time.perf_counter()
    try:
        det=detect.run(cfg,clip,model=SourceGeometryDetector(engine,(clip.info.width,clip.info.height)),progress=False)
    finally:detect.iter_frames=original_decoder
    detection_seconds=time.perf_counter()-started
    if abs(len(det)/cfg.sample_fps-duration)>max(.25,1/cfg.sample_fps+.1):raise ValueError('Bounded decode clock differs')
    region=_region_for(cfg,clips,source)
    reference_clip=next(c for c in clips if str(c.path.resolve())==region['ref_file'] or c.path.name==region['ref_file'])
    reference_shape=scaled.resized_shape(reference_clip.info.width,reference_clip.info.height,640)
    with closing(media.iter_frames(reference_clip.path,1000,*reference_shape,start=region['ref_time'],duration=.2,hwaccel=args.backend=='cuda')) as frames:
        reference=next(frames)
    keyframes=[]
    for index,rgb in sorted(captured.items()):
        polygon,quality=project_field_region(reference,rgb,region['polygon'])
        keyframes.append(dict(at=clip.start+index/cfg.sample_fps,polygon=polygon,**quality))
    fallback=np.load(args.snapshot/record['name']/(source.path.stem+'-mask.npy'),allow_pickle=False)
    det=regions.dynamic_region_detections(det,keyframes,fallback,cfg)
    factory=regions.TimeScaledTracker if specification.get('tracker_policy')=='time-scaled' else None
    f=features.compute(cfg,det,np.ones_like(fallback),tracker_factory=factory)
    plays=[]
    if classifier is not None:
        prob=model.predict(classifier,f,cfg);plays=model.intervals(cfg,f.t,prob)
        plays=post.filter_by_peak(f.t,prob,plays,best.get('peak_threshold',0))
        plays=post.refine_boundaries(f.t,prob,plays,best.get('refine_fraction',0),best.get('refine_padding',0))
    body_seconds=time.perf_counter()-started
    after=source.path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during benchmark')
    row=dict(game=record['name'],source_offset=offset,video_seconds=duration,frames=len(f.t),decode_detection_seconds=detection_seconds,
        pipeline_seconds=body_seconds,realtime=duration/body_seconds,predicted_plays=len(plays),camera_projections_accepted=sum(k['accepted'] for k in keyframes))
    if classifier is None:row.pop("predicted_plays")
    return f,row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--size',type=int,default=1280)
    parser.add_argument('--backend',choices=('openvino','cuda'),required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--duration',type=float,default=180.)
    parser.add_argument('--games',type=int,default=3)
    parser.add_argument('--active-window',action='store_true',help='choose development timing windows around a GT play; never an accuracy evaluation')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new benchmark directory')
    if args.games<1 or args.duration<=0 or not np.isfinite(args.duration) or args.threads<1 or args.size<64 or args.size%32:raise ValueError('Invalid benchmark limits')
    if args.backend=='openvino':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    args.out.mkdir(parents=True)
    for variable,folder in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib'),('CUDA_CACHE_PATH','cuda')]:
        destination=args.out/'runtime'/folder;destination.mkdir(parents=True)
        os.environ[variable]=str(destination)
    regions=helper('prepare-camera-region-features');scaled=helper('prepare-scaled-detector-features');post=helper('calibrate-play-thresholds')
    candidate=next(c for c in json.loads(args.calibration.read_text())['candidates'] if c['id']==args.candidate)
    cfg=Config(**candidate['configuration']);best=candidate['best']
    cfg.model_threshold=best['threshold'];cfg.model_smooth=best['smooth'];cfg.model_min_duration=best['minimum_duration']
    specification=json.loads((args.features/'features.json').read_text())
    if specification['names']!=features.FEATURE_NAMES or specification.get('region_policy')!='dynamic':raise ValueError('Benchmark requires dynamic person features')
    if 'exclude' in specification:cfg.model_exclude=tuple(specification['exclude'])
    records=json.loads((args.snapshot/'dataset.json').read_text())
    for record in records:validate_name(record['name'])
    feature_records=json.loads((args.features/'dataset.json').read_text())
    identity=lambda rows:[{k:v for k,v in row.items() if k!='cache_provenance'} for row in rows]
    if identity(records)!=identity(feature_records):raise ValueError('Benchmark feature split/source identity differs')
    development=[r for r in records if r['partition']=='development'];training=[]
    for record in development:
        with np.load(args.features/record['name']/f'features-{cfg.sample_fps:g}.npz',allow_pickle=False) as z:
            if z['names'].tolist()!=features.FEATURE_NAMES:raise ValueError('Feature names differ')
            f=features.Features(z['t'],z['X'],features.FEATURE_NAMES)
        training.append((record['name'],f,[Marked(i,a,b) for i,(a,b) in enumerate(record['marks'],1)]))
    started=time.perf_counter();classifier=model.fit(cfg,model.build(cfg,training));training_seconds=time.perf_counter()-started
    cfg.model=str(args.weights);cfg.imgsz=args.size;cfg.inference_cache_dir=str(args.out/'exports');cfg.inference_threads=args.threads
    cfg.detector_backend='openvino' if args.backend=='openvino' else 'torch';cfg.device='cpu' if args.backend=='openvino' else 'cuda:0';cfg.batch=1 if args.backend=='openvino' else 16
    started=time.perf_counter()
    if args.backend=='openvino':cfg.model=str(inference.prepare_graph(cfg))
    export_seconds=time.perf_counter()-started
    started=time.perf_counter();engine=detect.create_detector(cfg);load_seconds=time.perf_counter()-started
    devices=None
    if args.backend=='openvino':
        devices=list(engine.engine.get_property('EXECUTION_DEVICES'))
        if not devices or any(not device.startswith('CPU') for device in devices):raise RuntimeError('Expected CPU-only execution')
    report=dict(scope='detector loading, bounded source decode, dynamic camera projections, tracking, features, classifier and postprocessing; excludes imports, source probing, head training and first graph export',
        accuracy_scope='development selection and timing only; benchmark games are training data, not an independent quality test',
        candidate=args.candidate,development_quality=best['event'],development_minimum_game_recall=best['minimum_game_recall'],
        backend=args.backend,cpu_execution_devices=devices,threads=args.threads,imgsz=args.size,sample_fps=cfg.sample_fps,
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),training_seconds=training_seconds,
        first_export_seconds=export_seconds,detector_load_seconds=load_seconds,cases=[])
    report.update(active_window=args.active_window,feature_schema=specification['schema'],
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    original_decoder=detect.iter_frames
    try:
        for record in development[:args.games]:
            cfg.fields_dir=str(args.snapshot/record['name']/'fields');cfg.cache_dir=str(args.out/'detections'/record['name'])
            clips=groundtruth.project_clips(args.snapshot/record['name']/'review.mlt');source=clips[0];identity=record['clips'][0]
            duration=min(args.duration,source.info.duration)
            offset=active_window_start(record['marks'],source.start,source.info.duration,duration) if args.active_window else 0.
            f,row=run_source(cfg,source,clips,record,identity,args,engine,regions,scaled,specification,
                duration,offset,classifier,best,post)
            report['cases'].append(row);print(json.dumps(row),flush=True)
            (args.out/'report.partial.json').write_text(json.dumps(report,indent=2))
    finally:detect.iter_frames=original_decoder
    report['total_video_seconds']=sum(r['video_seconds'] for r in report['cases'])
    report['total_seconds_including_detector_load']=load_seconds+sum(r['pipeline_seconds'] for r in report['cases'])
    report['realtime_including_detector_load']=report['total_video_seconds']/report['total_seconds_including_detector_load']
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
