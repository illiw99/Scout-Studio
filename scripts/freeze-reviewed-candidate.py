#!/usr/bin/env python3
"""Freeze a development-only numeric research head before reserved-game evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import classifier_runtime,features,model
from scout.config import Config
from scout.groundtruth import Marked
from scout.storage import validate_name


def dataset_identity(records):
    stable=[{k:v for k,v in r.items() if k!='cache_provenance'} for r in records]
    return hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_head(path,records,weights):
    payload=json.loads(path.read_text())
    if payload.get('format')!='scout-research-head-v1':raise ValueError('Expected a frozen research head')
    if payload['dataset_identity']!=dataset_identity(records):raise ValueError('Frozen split, GT or source identity changed')
    if (set(payload['training_games'])!={r['name'] for r in records if r['partition']=='development'}
            or set(payload['reserved_games'])!={r['name'] for r in records if r['partition']=='holdout'}):raise ValueError('Frozen training/reserved membership differs')
    if payload['weights_sha256']!=hashlib.sha256(weights.read_bytes()).hexdigest():raise ValueError('Detector weights differ from frozen head')
    cfg=Config(**payload['configuration']);core=payload['classifier']
    expected=model.window_names([n for n in features.FEATURE_NAMES if n not in set(cfg.model_exclude or ())])
    if (core['names']!=expected or core['offsets']!=list(model.OFFSETS) or core['sample_fps']!=cfg.sample_fps
            or core['exclude']!=list(cfg.model_exclude or ())):raise ValueError('Frozen classifier schema differs')
    return classifier_runtime.Classifier(core),cfg,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--size',type=int,default=1280)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Frozen heads are never overwritten')
    if args.size<64 or args.size%32:raise ValueError('Invalid detector size')
    candidate=next(c for c in json.loads(args.calibration.read_text())['candidates'] if c['id']==args.candidate)
    cfg=Config(**candidate['configuration']);best=candidate['best']
    cfg.model_threshold=best['threshold'];cfg.model_smooth=best['smooth'];cfg.model_min_duration=best['minimum_duration']
    records=json.loads((args.features/'dataset.json').read_text());spec=json.loads((args.features/'features.json').read_text())
    for record in records:
        validate_name(record['name'])
        if record['partition'] not in ('development','holdout'):raise ValueError('Unknown split')
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate games')
    if spec['names']!=features.FEATURE_NAMES or spec.get('region_policy')!='dynamic':raise ValueError('Expected dynamic person features')
    if 'exclude' in spec:cfg.model_exclude=tuple(spec['exclude'])
    train=[r for r in records if r['partition']=='development'];reserved=[r for r in records if r['partition']=='holdout']
    if not train or not reserved or {r['group'] for r in train}&{r['group'] for r in reserved}:raise ValueError('Reserve independent match days')
    cases=[];hashes={}
    for record in train:
        file=args.features/record['name']/f'features-{cfg.sample_fps:g}.npz'
        with np.load(file,allow_pickle=False) as z:
            if z['names'].tolist()!=features.FEATURE_NAMES:raise ValueError('Feature names differ')
            f=features.Features(z['t'],z['X'],features.FEATURE_NAMES)
        cases.append((record['name'],f,[Marked(i,a,b) for i,(a,b) in enumerate(record['marks'],1)]));hashes[record['name']]=hashlib.sha256(file.read_bytes()).hexdigest()
    dataset=model.build(cfg,cases);classifier=model.fit(cfg,dataset)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.freeze-',dir=args.out.parent) as directory:
        core=Path(directory)/'core.json';classifier_runtime.export(core,classifier,dataset.names,cfg,model.OFFSETS)
        payload=dict(format='scout-research-head-v1',candidate=args.candidate,configuration=asdict(cfg),pipeline=spec,
            classifier=json.loads(core.read_text()),dataset_identity=dataset_identity(records),training_features_sha256=hashes,
            training_games=[r['name'] for r in train],reserved_games=[r['name'] for r in reserved],imgsz=args.size,
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
            postprocessing={k:best[k] for k in ('peak_threshold','refine_fraction','refine_padding') if k in best},
            development_quality=best['event'],development_minimum_game_recall=best['minimum_game_recall'])
        temporary=Path(directory)/'head.json';temporary.write_text(json.dumps(payload,separators=(',',':')))
        load_head(temporary,records,args.weights)
        os.link(temporary,args.out)
    print('Frozen numeric head; reserved arrays were not read',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
