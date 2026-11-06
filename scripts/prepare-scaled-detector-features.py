#!/usr/bin/env python3
"""Isolated known-weight detector experiment with resizing before RGB transfer.

No production defaults or caches are changed. Requires CUDA-capable FFmpeg;
failure stops the experiment rather than silently changing its backend.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import detect,features,groundtruth,media
from scout.config import Config
from scout.model_evaluation import subsample
from scout.storage import validate_name


def resized_shape(width,height,size):
    if min(width,height,size)<=0:raise ValueError('Positive frame dimensions are required')
    ratio=min(1.,size/max(width,height))
    return max(2,int(width*ratio)//2*2),max(2,int(height*ratio)//2*2)


def source_boxes(boxes,decoded_shape,source_shape):
    w,h=decoded_shape;original_w,original_h=source_shape
    return np.asarray(boxes,np.float32)*np.asarray([original_w/w,original_h/h,original_w/w,original_h/h],np.float32)


def cuda_frames(path,fps,width,height,duration=None,start=0.):
    cmd=[media.ffmpeg_exe(),'-hide_banner','-loglevel','error','-hwaccel','cuda',
         '-hwaccel_output_format','cuda']
    if start:cmd+=['-ss',str(start)]
    cmd+=['-i',str(path)]
    if duration is not None:cmd+=['-t',str(duration)]
    cmd+=['-vf',
         f'fps={fps:g},scale_cuda={width}:{height},hwdownload,format=nv12,format=rgb24',
         '-f','rawvideo','-']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    drained=False
    try:
        size=width*height*3
        while True:
            data=bytearray()
            while len(data)<size:
                block=proc.stdout.read(size-len(data))
                if not block:break
                data.extend(block)
            if not data:drained=True;break
            if len(data)!=size:raise RuntimeError('Incomplete CUDA-decoded RGB frame')
            yield np.frombuffer(data,np.uint8).reshape(height,width,3)
    finally:
        if not drained and proc.poll() is None:proc.kill()
        proc.stdout.close();error=proc.stderr.read().decode('utf-8','replace');proc.stderr.close()
        status=proc.wait()
        if drained and status:raise RuntimeError('CUDA scaling failed: '+error[-1500:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--size',type=int,default=640)
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new experiment directory')
    if args.size<64 or args.size%32 or args.batch<1:raise ValueError('Use a positive batch and image size divisible by32')
    weight_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    records=json.loads((args.snapshot/'dataset.json').read_text())
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate games')
    for record in records:
        validate_name(record['name'])
        if record['partition'] not in ('development','holdout'):raise ValueError('Unknown split')
    runtime=args.out.parent/'.scout-experiment-runtime'
    for variable,folder in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib'),('CUDA_CACHE_PATH','cuda')]:
        destination=runtime/folder;destination.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.environ[variable]=str(destination)
    import cv2
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required by this experiment')
    torch.set_num_threads(4)
    detector=YOLO(str(args.weights))
    args.out.mkdir(parents=True)
    provenance=dict(weights_sha256=weight_hash,size=args.size,batch=args.batch,source_fps=5,
        decoder='CUDA decode, scale_cuda bilinear, NV12 download, RGB24 conversion',
        boxes='mapped back to source geometry before field filtering/tracking',torch=torch.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    records=[dict(record,cache_provenance=f'fresh {args.weights.name}; pre-scaled CUDA decode; weight sha256={weight_hash}') for record in records]
    (args.out/'dataset.json').write_text(json.dumps(records,indent=2))
    (args.out/'features.json').write_text(json.dumps(dict(schema='scout-person-prescaled-v1',names=features.FEATURE_NAMES,
        rates=[5.,2.5,1.],scope='fresh known-weight pre-scaled CUDA extraction; development only; not a production feature schema'),indent=2))
    for record in records:
        if record['partition']!='development':continue
        game=record['name'];folder=args.out/game;folder.mkdir();cache=folder/'detections';cache.mkdir()
        clips=groundtruth.project_clips(args.snapshot/game/'review.mlt')
        if len(clips)!=len(record['clips']):raise ValueError('Source count changed')
        parts={fps:[] for fps in (5.,2.5,1.)}
        for clip,identity in zip(clips,record['clips']):
            before=clip.path.stat()
            if clip.path.name!=Path(identity['source']).name or (before.st_size,before.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Source identity changed')
            shape=resized_shape(clip.info.width,clip.info.height,args.size);batch=[];boxes=[];conf=[];offsets=[0];start=time.perf_counter()
            def flush():
                if not batch:return
                results=detector.predict(batch,imgsz=args.size,conf=Config().conf,classes=[0],device=0,
                    verbose=False,**detect._precision_kwargs(Config(),0))
                if len(results)!=len(batch):raise ValueError('Detector result count differs from batch')
                for result in results:
                    b=source_boxes(result.boxes.xyxy.cpu().numpy(),shape,(clip.info.width,clip.info.height))
                    boxes.append(b);conf.append(result.boxes.conf.cpu().numpy());offsets.append(offsets[-1]+len(b))
                batch.clear()
            frames=cuda_frames(clip.path,5,*shape)
            try:
                for rgb in frames:
                    batch.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    if len(batch)>=args.batch:flush()
                flush()
            finally:frames.close()
            elapsed=time.perf_counter()-start;n=len(offsets)-1
            if abs(n/5-clip.info.duration)>.25:raise ValueError('Decoded source timing differs')
            det=detect.Detections(clip.start+np.arange(n)/5,np.asarray(offsets,np.int64),np.concatenate(boxes),np.concatenate(conf),(clip.info.height,clip.info.width))
            target=cache/(clip.path.stem+'.prescaled.npz');temporary=target.with_suffix('.partial.npz')
            np.savez_compressed(temporary,times=det.times,offsets=det.offsets,boxes=det.boxes,conf=det.conf,mask_shape=np.asarray(det.mask_shape));temporary.replace(target)
            mask=np.load(args.snapshot/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
            cfg=Config();feature_start=time.perf_counter()
            for fps in parts:
                cfg.sample_fps=fps;parts[fps].append(features.compute(cfg,subsample(det,5,fps),mask))
            after=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during extraction')
            result=dict(game=game,source=identity['source'],frames=n,decoded_shape=shape,source_seconds=clip.info.duration,
                detection_seconds=elapsed,three_rate_feature_seconds=time.perf_counter()-feature_start,
                scope='shared experimental extraction timing; not isolated end-to-end benchmark')
            (folder/(clip.path.stem+'-timing.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        for fps,items in parts.items():
            np.savez_compressed(folder/f'features-{fps:g}.npz',t=np.concatenate([f.t for f in items]),X=np.concatenate([f.X for f in items]),names=np.asarray(features.FEATURE_NAMES))
        print(game,'pre-scaled detector features ready',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
