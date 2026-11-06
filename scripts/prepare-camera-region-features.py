#!/usr/bin/env python3
"""Experimental camera-region unions or full-frame person features.

Preserves every source polygon. Reads a frozen private snapshot, writes a new
private feature dataset and does not process reserved games by default.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import math
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import detect,features,field,groundtruth
from scout.config import Config
from scout.cli import _region_for
from scout.media import grab_frame
from scout.model_evaluation import project_field_region,subsample
from scout.storage import validate_name
from scout.track import Tracker


class TimeScaledTracker(Tracker):
    """Research ablation: preserve the 5 Hz settings in units of elapsed time."""
    def __init__(self,dt):
        super().__init__(dt=dt,gate=1.5*max(1.,5*dt),smooth=1-.4**(5*dt),
                         max_missed=max(1,math.ceil(.6/dt-1e-9)))
        self.required=max(1,math.ceil(.4/dt-1e-9))

    def update(self,pts,hs):
        live=super().update(pts,hs)
        # Features require age>=2. Adapt that view without mutating track ages.
        return [replace(track,age=2) if self.required<=track.age<2 else track for track in live]


def dynamic_region_detections(det,keyframes,fallback,cfg):
    """Filter source-coordinate boxes with an interpolated, time-local polygon."""
    accepted=sorted((k['at'],k['polygon']) for k in keyframes if k.get('polygon') is not None)
    times=np.asarray([k[0] for k in accepted]);points=np.asarray([k[1] for k in accepted])
    boxes=[];conf=[];offsets=[0]
    for i,when in enumerate(det.times):
        if len(accepted):
            coordinates=np.asarray([np.interp(when,times,points[:,vertex,axis])
                for vertex in range(points.shape[1]) for axis in range(2)]).reshape(-1,2)
            mask=field.polygon_mask(coordinates,fallback.shape,cfg.field_shrink_top)
        else:mask=fallback
        b,c=det.frame(i);keep=features._on_field(b,mask,cfg)
        boxes.append(b[keep]);conf.append(c[keep]);offsets.append(offsets[-1]+int(keep.sum()))
    return detect.Detections(det.times.copy(),np.asarray(offsets,np.int64),np.concatenate(boxes),np.concatenate(conf),det.mask_shape)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--detections-from',type=Path,help='optional fresh-detector dataset')
    parser.add_argument('--masks-from',type=Path,help='reuse completed masks from this exact snapshot')
    parser.add_argument('--include-holdout',action='store_true')
    parser.add_argument('--region-policy',choices=('camera-union','full-frame','dynamic'),default='camera-union')
    parser.add_argument('--min-person-height',type=float,help='experimental minimum height as fraction of frame height')
    parser.add_argument('--rates',type=float,nargs='+',default=[5.,2.5,1.])
    parser.add_argument('--time-scaled-tracker',action='store_true',help='experimental temporal gates; keeps the 5Hz baseline unchanged')
    args=parser.parse_args()
    if args.min_person_height is not None and not 0<args.min_person_height<Config().max_person_h:raise ValueError('Invalid minimum person height')
    if args.masks_from and args.region_policy!='camera-union':raise ValueError('Reused masks require camera-union policy')
    if not set(args.rates)<={5.,2.5,1.} or len(set(args.rates))!=len(args.rates):raise ValueError('Use unique supported rates:5,2.5,1')
    if args.out.exists():raise ValueError('Choose a new experiment directory')
    records=json.loads((args.snapshot/'dataset.json').read_text())
    for record in records:validate_name(record['name'])
    args.out.mkdir(parents=True)
    (args.out/'dataset.json').write_text(json.dumps(records,indent=2))
    (args.out/'features.json').write_text(json.dumps(dict(schema='scout-person-camera-union-v1',
        names=features.FEATURE_NAMES,rates=args.rates,region_policy=args.region_policy,min_person_height=args.min_person_height,
        tracker_policy='time-scaled' if args.time_scaled_tracker else 'reference',
        scope='experimental label-independent person region/height variant; not a production feature schema'),indent=2))
    for record in records:
        if record['partition']!='development' and not args.include_holdout:continue
        game=record['name'];private=args.out/game;private.mkdir()
        clips=groundtruth.project_clips(args.snapshot/game/'review.mlt')
        if len(clips)!=len(record['clips']):raise ValueError('Snapshot source count changed')
        cfg=Config(fields_dir=str(args.snapshot/game/'fields'),out_dir=str(private/'output'),cache_dir=str(private/'unused'))
        if args.min_person_height is not None:cfg.min_person_h=args.min_person_height
        parts={fps:[] for fps in args.rates};quality=[]
        for clip,identity in zip(clips,record['clips']):
            before=clip.path.stat()
            if clip.path.name!=Path(identity['source']).name or (before.st_size,before.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Source identity/order changed')
            start=time.perf_counter()
            legacy=np.load(args.snapshot/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
            if args.region_policy=='full-frame':
                mask=np.ones_like(legacy);accepted=[]
            elif args.masks_from:
                original=json.loads((args.masks_from/game/'region-provenance.json').read_text())
                if original['source_clips']!=record['clips']:raise ValueError('Region source identity differs')
                if original.get('region_policy','camera-union')!='camera-union':raise ValueError('Only camera-union masks can be reused')
                mask=np.load(args.masks_from/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
                accepted=[]
            else:
                region=_region_for(cfg,clips,clip)
                if region is None:raise ValueError('No copied field region')
                mask=legacy.copy();reference=None
                refclip=next((c for c in clips if str(c.path.resolve())==region['ref_file'] or c.path.name==region['ref_file']),None)
                if refclip:
                    try:reference=grab_frame(refclip.path,region['ref_time'])
                    except Exception:pass
                accepted=[]
                if reference is not None:
                    mask|=field.polygon_mask(region['polygon'],mask.shape,cfg.field_shrink_top)
                    fractions=np.array([0.,.15,.325,.5,.675,.85,1.]) if args.region_policy=='dynamic' else np.linspace(.15,.85,5)
                    for when in np.clip(fractions*clip.info.duration,0,max(0,clip.info.duration-.1)):
                        image=grab_frame(clip.path,float(when))
                        projected,info=project_field_region(reference,image,region['polygon'])
                        info['at']=float(when);info['polygon']=projected;accepted.append(info)
                        if projected is not None:mask|=field.polygon_mask(projected,mask.shape,cfg.field_shrink_top)
            np.save(private/(clip.path.stem+'-mask.npy'),mask)
            folder=(args.detections_from/game/'detections') if args.detections_from else (args.snapshot/game/'cache')
            files=list(folder.glob(clip.path.stem+'.*.npz'))
            if len(files)!=1:raise ValueError('Use one unambiguous frozen detection cache per source')
            with np.load(files[0],allow_pickle=False) as z:
                det=detect.Detections(clip.start+np.arange(len(z['times']))/5,z['offsets'],z['boxes'],z['conf'],tuple(z['mask_shape']))
            if abs(len(det)/5-clip.info.duration)>.25 or det.mask_shape!=mask.shape:raise ValueError('Detection timing/geometry mismatch')
            if args.region_policy=='dynamic':
                absolute=[dict(k,at=k['at']+clip.start) for k in accepted]
                det=dynamic_region_detections(det,absolute,legacy,cfg)
                mask=np.ones_like(legacy)
            for fps in parts:
                cfg.sample_fps=fps;parts[fps].append(features.compute(cfg,subsample(det,5,fps),mask,
                    tracker_factory=TimeScaledTracker if args.time_scaled_tracker else None))
            after=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during extraction')
            quality.append(dict(source=identity['source'],keyframes=accepted,legacy_area=int(legacy.sum()),union_area=int(mask.sum()),seconds=time.perf_counter()-start))
        for fps,items in parts.items():np.savez_compressed(private/f'features-{fps:g}.npz',t=np.concatenate([f.t for f in items]),X=np.concatenate([f.X for f in items]),names=np.asarray(items[0].names))
        (private/'region-provenance.json').write_text(json.dumps(dict(source_clips=record['clips'],quality=quality,
            region_policy=args.region_policy,min_person_height=cfg.min_person_h,
            algorithm=('full-frame' if args.region_policy=='full-frame' else
                'ORB/RANSAC projection at seven fixed times; interpolate polygons per frame; legacy fallback if none accepted'
                if args.region_policy=='dynamic' else 'ORB/RANSAC reference projection at five fixed times; union with legacy region')),indent=2))
        print(game,args.region_policy,'features ready',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
