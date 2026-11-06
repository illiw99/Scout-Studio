#!/usr/bin/env python3
"""Experimental camera-region unions or full-frame person features.

Preserves every source polygon. Reads a frozen private snapshot, writes a new
private feature dataset and does not process reserved games by default.
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import detect,features,field,groundtruth
from scout.config import Config
from scout.cli import _region_for
from scout.media import grab_frame
from scout.model_evaluation import project_field_region,subsample
from scout.storage import validate_name
from scout.track import TimeScaledTracker
from scout.camera import filter_detections as dynamic_region_detections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--detections-from',type=Path,help='optional fresh-detector dataset')
    parser.add_argument('--masks-from',type=Path,help='reuse completed masks from this exact snapshot')
    parser.add_argument('--dynamic-from',type=Path,help='reuse dynamic keyframe polygons from this exact snapshot')
    parser.add_argument('--include-holdout',action='store_true')
    parser.add_argument('--region-policy',choices=('camera-union','full-frame','dynamic'),default='camera-union')
    parser.add_argument('--min-person-height',type=float,help='experimental minimum height as fraction of frame height')
    parser.add_argument('--rates',type=float,nargs='+',default=[5.,2.5,1.])
    parser.add_argument('--time-scaled-tracker',action='store_true',help='experimental temporal gates; keeps the 5Hz baseline unchanged')
    args=parser.parse_args()
    if args.min_person_height is not None and not 0<args.min_person_height<Config().max_person_h:raise ValueError('Invalid minimum person height')
    if args.masks_from and args.region_policy!='camera-union':raise ValueError('Reused masks require camera-union policy')
    if args.dynamic_from and (args.region_policy!='dynamic' or args.masks_from):raise ValueError('Reused dynamic polygons require dynamic policy')
    if not set(args.rates)<={5.,2.5,1.} or len(set(args.rates))!=len(args.rates):raise ValueError('Use unique supported rates:5,2.5,1')
    if args.out.exists():raise ValueError('Choose a new experiment directory')
    records=json.loads((args.snapshot/'dataset.json').read_text())
    for record in records:validate_name(record['name'])
    if args.detections_from:
        detector_records=json.loads((args.detections_from/'dataset.json').read_text())
        identity=lambda rows:[{k:v for k,v in row.items() if k!='cache_provenance'} for row in rows]
        if identity(records)!=identity(detector_records):raise ValueError('Detector dataset source identities, labels or splits differ')
        records=[dict(record,cache_provenance=source.get('cache_provenance','unverified')) for record,source in zip(records,detector_records)]
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
            if args.dynamic_from:
                original=json.loads((args.dynamic_from/game/'region-provenance.json').read_text())
                if original['source_clips']!=record['clips'] or original.get('region_policy')!='dynamic':raise ValueError('Dynamic region source identity/policy differs')
                entry=next(q for q in original['quality'] if q['source']==identity['source'])
                accepted=entry['keyframes']
                mask=np.load(args.dynamic_from/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
            elif args.region_policy=='full-frame':
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
            det_fps=5.
            if args.region_policy=='dynamic':
                if len(parts)==1:
                    det_fps=next(iter(parts));det=subsample(det,5,det_fps)
                absolute=[dict(k,at=k['at']+clip.start) for k in accepted]
                det=dynamic_region_detections(det,absolute,legacy,cfg)
                mask=np.ones_like(legacy)
            for fps in parts:
                cfg.sample_fps=fps;parts[fps].append(features.compute(cfg,subsample(det,det_fps,fps),mask,
                    tracker_factory=TimeScaledTracker if args.time_scaled_tracker else None))
            after=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during extraction')
            quality.append(dict(source=identity['source'],input_cache_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
                keyframes=accepted,legacy_area=int(legacy.sum()),union_area=int(mask.sum()),seconds=time.perf_counter()-start))
        for fps,items in parts.items():np.savez_compressed(private/f'features-{fps:g}.npz',t=np.concatenate([f.t for f in items]),X=np.concatenate([f.X for f in items]),names=np.asarray(items[0].names))
        (private/'region-provenance.json').write_text(json.dumps(dict(source_clips=record['clips'],quality=quality,
            region_policy=args.region_policy,min_person_height=cfg.min_person_h,
            algorithm=('full-frame' if args.region_policy=='full-frame' else
                'ORB/RANSAC projection at seven fixed times; interpolate polygons per frame; legacy fallback if none accepted'
                if args.region_policy=='dynamic' else 'ORB/RANSAC reference projection at five fixed times; union with legacy region')),indent=2))
        print(game,args.region_policy,'features ready',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
