#!/usr/bin/env python3
"""Experimental field-region unions from label-independent camera keyframes.

Preserves every source polygon. Reads a frozen private snapshot, writes a new
private feature dataset and does not process reserved games by default.
"""
import argparse
import json
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--detections-from',type=Path,help='optional fresh-detector dataset')
    parser.add_argument('--masks-from',type=Path,help='reuse completed masks from this exact snapshot')
    parser.add_argument('--include-holdout',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new experiment directory')
    records=json.loads((args.snapshot/'dataset.json').read_text())
    for record in records:validate_name(record['name'])
    args.out.mkdir(parents=True)
    (args.out/'dataset.json').write_text(json.dumps(records,indent=2))
    (args.out/'features.json').write_text(json.dumps(dict(schema='scout-person-camera-union-v1',
        names=features.FEATURE_NAMES,rates=[5.,2.5,1.],scope='experimental label-independent camera-region union; not a production feature schema'),indent=2))
    for record in records:
        if record['partition']!='development' and not args.include_holdout:continue
        game=record['name'];private=args.out/game;private.mkdir()
        clips=groundtruth.project_clips(args.snapshot/game/'review.mlt')
        if len(clips)!=len(record['clips']):raise ValueError('Snapshot source count changed')
        cfg=Config(fields_dir=str(args.snapshot/game/'fields'),out_dir=str(private/'output'),cache_dir=str(private/'unused'))
        parts={fps:[] for fps in (5.,2.5,1.)};quality=[]
        for clip,identity in zip(clips,record['clips']):
            before=clip.path.stat()
            if clip.path.name!=Path(identity['source']).name or (before.st_size,before.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Source identity/order changed')
            start=time.perf_counter()
            legacy=np.load(args.snapshot/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
            if args.masks_from:
                original=json.loads((args.masks_from/game/'region-provenance.json').read_text())
                if original['source_clips']!=record['clips']:raise ValueError('Region source identity differs')
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
                    for when in np.linspace(.15,.85,5)*clip.info.duration:
                        image=grab_frame(clip.path,float(when))
                        projected,info=project_field_region(reference,image,region['polygon'])
                        info['at']=float(when);accepted.append(info)
                        if projected is not None:mask|=field.polygon_mask(projected,mask.shape,cfg.field_shrink_top)
            np.save(private/(clip.path.stem+'-mask.npy'),mask)
            folder=(args.detections_from/game/'detections') if args.detections_from else (args.snapshot/game/'cache')
            files=list(folder.glob(clip.path.stem+'.*.npz'))
            if len(files)!=1:raise ValueError('Use one unambiguous frozen detection cache per source')
            with np.load(files[0],allow_pickle=False) as z:
                det=detect.Detections(clip.start+np.arange(len(z['times']))/5,z['offsets'],z['boxes'],z['conf'],tuple(z['mask_shape']))
            if abs(len(det)/5-clip.info.duration)>.25 or det.mask_shape!=mask.shape:raise ValueError('Detection timing/geometry mismatch')
            for fps in parts:
                cfg.sample_fps=fps;parts[fps].append(features.compute(cfg,subsample(det,5,fps),mask))
            after=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during extraction')
            quality.append(dict(source=identity['source'],keyframes=accepted,legacy_area=int(legacy.sum()),union_area=int(mask.sum()),seconds=time.perf_counter()-start))
        for fps,items in parts.items():np.savez_compressed(private/f'features-{fps:g}.npz',t=np.concatenate([f.t for f in items]),X=np.concatenate([f.X for f in items]),names=np.asarray(items[0].names))
        (private/'region-provenance.json').write_text(json.dumps(dict(source_clips=record['clips'],quality=quality,algorithm='ORB/RANSAC reference projection at five fixed times; union with legacy region'),indent=2))
        print(game,'camera-union features ready',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
