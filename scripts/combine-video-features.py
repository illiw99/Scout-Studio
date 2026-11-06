#!/usr/bin/env python3
"""Fuse frozen development features without decoding sources or reading holdout arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.storage import validate_name
from scout.config import Config


def align_nearest(target, source, maximum_distance):
    """Nearest timestamp, preferring the past on ties; never silently extrapolate."""
    target=np.asarray(target);source=np.asarray(source)
    if (target.ndim!=1 or source.ndim!=1 or not len(target) or not len(source)
            or not np.isfinite(target).all() or not np.isfinite(source).all()
            or np.any(np.diff(target)<=0) or np.any(np.diff(source)<=0)
            or not np.isfinite(maximum_distance) or maximum_distance<=0):
        raise ValueError('Ordered finite clocks and a positive alignment limit are required')
    right=np.clip(np.searchsorted(source,target),0,len(source)-1)
    left=np.maximum(0,right-1)
    index=np.where(np.abs(target-source[left])<=np.abs(target-source[right]),left,right)
    if np.any(np.abs(target-source[index])>maximum_distance):
        raise ValueError('Feature clocks differ beyond the alignment limit')
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left',type=Path,required=True)
    parser.add_argument('--right',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--left-fps',type=float,default=5.)
    parser.add_argument('--right-fps',type=float,default=1.)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new fusion directory')
    if any(not np.isfinite(fps) or fps<=0 for fps in (args.left_fps,args.right_fps)):raise ValueError('Invalid feature rate')
    datasets=[json.loads((root/'dataset.json').read_text()) for root in (args.left,args.right)]
    identities=[[{k:v for k,v in record.items() if k!='cache_provenance'} for record in records] for records in datasets]
    if identities[0]!=identities[1]:raise ValueError('Feature datasets must use identical source identities, GT and splits')
    records=datasets[0]
    for record in records:
        validate_name(record['name'])
        if record['partition'] not in ('development','holdout'):raise ValueError('Unknown split')
    specifications=[json.loads((root/'features.json').read_text()) for root in (args.left,args.right)]
    names=[prefix+name for prefix,spec in zip(('left__','right__'),specifications) for name in spec['names']]
    exclude=[prefix+name for prefix,spec in zip(('left__','right__'),specifications)
             for name in spec.get('exclude',Config().model_exclude if spec['schema'].startswith('scout-person') else [])]
    args.out.mkdir(parents=True)
    (args.out/'dataset.json').write_text(json.dumps(records,indent=2))
    spec=dict(schema='scout-feature-fusion-v1',names=names,exclude=exclude,rates=[args.left_fps],
              scope='experimental development feature fusion; nearest right-clock sample; excludes reserved arrays',
              source_schemas=[spec['schema'] for spec in specifications],right_fps=args.right_fps,
              source_dataset_sha256=[hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest() for root in (args.left,args.right)])
    (args.out/'features.json').write_text(json.dumps(spec,indent=2))
    provenance=[]
    for record in records:
        if record['partition']!='development':continue
        arrays=[];clocks=[];identities=[]
        for root,fps,source_spec in zip((args.left,args.right),(args.left_fps,args.right_fps),specifications):
            file=root/record['name']/f'features-{fps:g}.npz'
            with np.load(file,allow_pickle=False) as data:
                t,X,actual_names=data['t'],data['X'],data['names'].tolist()
            if actual_names!=source_spec['names'] or X.shape!=(len(t),len(actual_names)) or not np.isfinite(X).all():raise ValueError('Invalid feature matrix')
            clocks.append(t);arrays.append(X);identities.append(hashlib.sha256(file.read_bytes()).hexdigest())
        index=align_nearest(clocks[0],clocks[1],1/args.right_fps+.001)
        folder=args.out/record['name'];folder.mkdir()
        np.savez_compressed(folder/f'features-{args.left_fps:g}.npz',t=clocks[0],X=np.concatenate([arrays[0],arrays[1][index]],axis=1),names=np.asarray(names))
        provenance.append(dict(game=record['name'],source_sha256=identities,maximum_clock_distance=float(np.max(np.abs(clocks[0]-clocks[1][index])))))
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
