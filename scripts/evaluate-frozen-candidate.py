#!/usr/bin/env python3
"""Evaluate a fixed research pipeline on reserved full games without training/tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import detect,features,groundtruth,inference,model
from scout.model_evaluation import interval_metrics,summarize
from scout.storage import validate_name
import importlib.util


def helper(filename):
    spec=importlib.util.spec_from_file_location(filename.replace('-','_'),Path(__file__).with_name(filename+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--head',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--backend',choices=('cuda','openvino'),required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--export-cache',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new confirmation directory')
    if args.threads<1:raise ValueError('Positive CPU thread count required')
    bench=helper('benchmark-reviewed-pipeline');frozen=helper('freeze-reviewed-candidate')
    regions=helper('prepare-camera-region-features');scaled=helper('prepare-scaled-detector-features');post=helper('calibrate-play-thresholds')
    records=json.loads((args.snapshot/'dataset.json').read_text());started=time.perf_counter()
    for record in records:validate_name(record['name'])
    classifier,cfg,head=frozen.load_head(args.head,records,args.weights);head_load_seconds=time.perf_counter()-started
    reserved=[r for r in records if r['partition']=='holdout']
    if {r['name'] for r in reserved}!=set(head['reserved_games']) or set(head['training_games'])&{r['name'] for r in reserved}:raise ValueError('Reserved game set differs from frozen selection')
    if args.backend=='openvino':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    args.out.mkdir(parents=True)
    for variable,folder in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib'),('CUDA_CACHE_PATH','cuda')]:
        destination=args.out/'runtime'/folder;destination.mkdir(parents=True);os.environ[variable]=str(destination)
    args.size=head['imgsz'];cfg.model=str(args.weights);cfg.imgsz=args.size;cfg.inference_threads=args.threads
    cfg.inference_cache_dir=str(args.export_cache or args.out/'exports')
    cfg.detector_backend='openvino' if args.backend=='openvino' else 'torch';cfg.device='cpu' if args.backend=='openvino' else 'cuda:0';cfg.batch=1 if args.backend=='openvino' else 16
    started=time.perf_counter()
    if args.backend=='openvino':cfg.model=str(inference.prepare_graph(cfg))
    export_seconds=time.perf_counter()-started
    started=time.perf_counter();engine=detect.create_detector(cfg);detector_load_seconds=time.perf_counter()-started
    devices=None
    if args.backend=='openvino':
        devices=list(engine.engine.get_property('EXECUTION_DEVICES'))
        if not devices or any(not d.startswith('CPU') for d in devices):raise RuntimeError('Expected CPU-only inference')
    report=dict(schema=1,scope='fixed candidate, reserved complete games; includes source probing, detector loading, decode, camera/track features, numeric head and postprocessing; first graph export excluded',
        baseline_disclosure='reserved GT was inspected for baseline reporting; never used to fit or tune this candidate',
        head_sha256=hashlib.sha256(args.head.read_bytes()).hexdigest(),candidate=head['candidate'],backend=args.backend,
        execution_devices=devices,head_load_seconds=head_load_seconds,detector_load_seconds=detector_load_seconds,
        first_export_seconds=export_seconds,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),games=[])
    for record in reserved:
        began=time.perf_counter();clips=groundtruth.project_clips(args.snapshot/record['name']/'review.mlt')
        if len(clips)!=len(record['clips']):raise ValueError('Reserved source count differs')
        cfg.fields_dir=str(args.snapshot/record['name']/'fields');cfg.cache_dir=str(args.out/'detections'/record['name'])
        folder=args.out/record['name'];folder.mkdir();parts=[];timings=[]
        for source,identity in zip(clips,record['clips']):
            if abs(source.start-identity['start'])>.01:raise ValueError('Reserved source ordering/offset changed')
            f,row=bench.run_source(cfg,source,clips,record,identity,args,engine,regions,scaled,head['pipeline'],source.info.duration)
            np.savez_compressed(folder/(source.path.stem+'-features.npz'),t=f.t,X=f.X,names=np.asarray(f.names))
            parts.append(f);timings.append(dict(source=source.path.name,**row));print(json.dumps(timings[-1]),flush=True)
            (folder/'progress.json').write_text(json.dumps(timings,indent=2))
        f=features.Features(np.concatenate([p.t for p in parts]),np.concatenate([p.X for p in parts]),features.FEATURE_NAMES)
        if np.any(np.diff(f.t)<=0):raise ValueError('Reserved full-game feature clock is not ordered')
        prob=model.predict(classifier,f,cfg);plays=model.intervals(cfg,f.t,prob);options=head['postprocessing']
        plays=post.filter_by_peak(f.t,prob,plays,options.get('peak_threshold',0))
        plays=post.refine_boundaries(f.t,prob,plays,options.get('refine_fraction',0),options.get('refine_padding',0))
        elapsed=time.perf_counter()-began;duration=sum(c.info.duration for c in clips)
        result=dict(game=record['name'],event=interval_metrics(record['marks'],plays,0),iou50=interval_metrics(record['marks'],plays,.5),
            predictions=plays,video_seconds=duration,pipeline_seconds=elapsed,realtime=duration/elapsed,source_timings=timings)
        report['games'].append(result);report['event']=summarize([g['event'] for g in report['games']]);report['iou50']=summarize([g['iou50'] for g in report['games']])
        (args.out/'report.partial.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(game=record['name'],event=summarize([result['event']]),realtime=result['realtime'])),flush=True)
    report['total_video_seconds']=sum(g['video_seconds'] for g in report['games'])
    report['total_seconds_including_loading']=head_load_seconds+detector_load_seconds+sum(g['pipeline_seconds'] for g in report['games'])
    report['realtime_including_loading']=report['total_video_seconds']/report['total_seconds_including_loading']
    report['minimum_game_recall']=min(summarize([g['event']])['recall'] for g in report['games'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
