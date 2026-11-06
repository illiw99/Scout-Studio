#!/usr/bin/env python3
"""Compare decision models using private, precomputed reviewed-game snapshots.

Never detects footage, edits reviews or reads the user's production model. Input
is dataset.json plus <game>/features-<fps>.npz prepared in an isolated workspace.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.config import Config
from scout import model_features
from scout.groundtruth import Marked
from scout import features, model
from scout.features import FEATURE_NAMES
from scout.model_evaluation import interval_metrics, summarize
from scout.storage import validate_name


def configuration(variant,fps):
    cfg=Config(sample_fps=fps)
    if variant=='compact-regularized':
        return replace(cfg,model_max_depth=3,model_max_iter=180,model_min_leaf=30,model_l2=3.)
    if variant in ('recall-weighted','linear-balanced'):
        return replace(cfg,model_class_weight='balanced')
    if variant!='reference-settings':raise ValueError('Unknown model variant')
    return cfg


def fit_candidate(cfg,ds,variant):
    if variant=='linear-balanced':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=500)).fit(ds.X,ds.y)
    return model.fit(cfg,ds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rates',nargs='+',type=float,help='feature sampling rates to compare')
    parser.add_argument('--variants',nargs='+',choices=('reference-settings','compact-regularized','recall-weighted','linear-balanced'),default=['reference-settings','compact-regularized','recall-weighted'])
    parser.add_argument('--minimum-recall',type=float,default=.95)
    parser.add_argument('--feature-set',choices=model_features.FEATURE_SETS,default=model_features.BASE)
    parser.add_argument('--context-offsets',type=float,nargs='+',default=list(model_features.DEFAULT_OFFSETS))
    parser.add_argument('--development-only',action='store_true',help='do not read/evaluate reserved game features')
    args=parser.parse_args()
    model_features.validate_offsets(args.context_offsets)
    if not 0 < args.minimum_recall <= 1:raise ValueError('Recall target must be in (0,1]')
    if args.out.exists() or args.out.with_suffix('.partial.json').exists():raise ValueError('Choose a new results file')
    records=json.loads((args.snapshot/'dataset.json').read_text())
    feature_file=args.snapshot/'features.json'
    specification=json.loads(feature_file.read_text()) if feature_file.exists() else dict(schema='scout-person-v1',names=FEATURE_NAMES,rates=[5.,2.5,1.])
    expected_names=specification['names']
    if not expected_names or len(set(expected_names))!=len(expected_names):raise ValueError('Invalid feature schema')
    rates=args.rates or specification['rates']
    if not rates or any(not np.isfinite(rate) or rate<=0 for rate in rates):raise ValueError('Invalid feature rate')
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate games')
    for r in records:
        validate_name(r['name'])
        if r['partition'] not in ('development','holdout') or not r['group']:raise ValueError('Invalid split')
    development={r['group'] for r in records if r['partition']=='development'}
    heldout={r['group'] for r in records if r['partition']=='holdout'}
    if len(development)<2 or not heldout or development&heldout:raise ValueError('Hold out independent match days')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema=2,scope=specification.get('scope','private reviewed-game feature screening; backend and source identity in snapshot'),
                probability_directory=args.out.stem+'.probabilities',
                selection='highest development precision meeting pooled and minimum-game event recall target; fallback maximum minimum-game recall; holdout never used to select/tune',
                minimum_event_recall=args.minimum_recall,stretch_event_recall=.98,feature_schema=specification['schema'],
                feature_set=args.feature_set,context_offsets=args.context_offsets,
                development_games=[r['name'] for r in records if r['partition']=='development'],
                holdout_games=[r['name'] for r in records if r['partition']=='holdout'],
                versions={name:version(name) for name in ('numpy','scikit-learn','scipy')},
                code_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                    for name in ('scout/model.py','scout/features.py','scout/model_evaluation.py','scripts/evaluate-reviewed-games.py')},
                candidates=[])
    loaded={}
    for fps in rates:
        cases=[]
        for r in records:
            if args.development_only and r['partition']=='holdout':continue
            with np.load(args.snapshot/r['name']/f'features-{fps:g}.npz',allow_pickle=False) as z:
                t,X,names=z['t'],z['X'],z['names'].tolist()
            if names!=expected_names or X.shape!=(len(t),len(names)) or not len(t) or np.isinf(X).any() or np.any(np.diff(t)<=0):
                raise ValueError('Incompatible feature snapshot')
            marks=[Marked(i,a,b) for i,(a,b) in enumerate(r['marks'],1)]
            cases.append((r,features.Features(t,X,names),marks))
        loaded[fps]=cases
        for variant in args.variants:
            cfg=configuration(variant,fps)
            if specification.get('region_policy')=='dynamic' and specification.get('tracker_policy')=='time-scaled':
                cfg.feature_pipeline='dynamic-field-v1'
            cfg.model_feature_set=args.feature_set;cfg.model_offsets=tuple(args.context_offsets)
            cfg.validate_pipeline()
            if 'exclude' in specification:cfg.model_exclude=tuple(specification['exclude'])
            result=dict(id=f'{variant}-{fps:g}fps',sample_fps=fps,variant=variant,configuration=asdict(cfg),
                        head_parameters=(dict(model='LogisticRegression',C=.1,class_weight='balanced',max_iter=500,standardize=True)
                                         if variant=='linear-balanced' else dict(model='HistGradientBoostingClassifier')),folds=[])
            start=time.perf_counter()
            for group in sorted(development):
                train=[(r['name'],f,m) for r,f,m in cases if r['partition']=='development' and r['group']!=group]
                ds=model.build(cfg,train);classifier=fit_candidate(cfg,ds,variant)
                for r,f,marks in cases:
                    if r['partition']!='development' or r['group']!=group:continue
                    prob=model.predict(classifier,f,cfg);pred=model.intervals(cfg,f.t,prob)
                    destination=args.out.parent/report['probability_directory']/result['id']
                    destination.mkdir(parents=True,exist_ok=True)
                    target=destination/(r['name']+'.npz')
                    if target.exists():raise ValueError('Probability snapshot already exists')
                    np.savez_compressed(target,t=f.t,prob=prob,truth=np.asarray(r['marks']))
                    result['folds'].append(dict(game=r['name'],held_out_day=group,
                        predictions=pred,event=interval_metrics(r['marks'],pred,0),
                        iou30=interval_metrics(r['marks'],pred,.3),iou50=interval_metrics(r['marks'],pred,.5)))
            result['event']=summarize([row['event'] for row in result['folds']])
            result['minimum_game_recall']=min(summarize([row['event']])['recall'] for row in result['folds'])
            result['meets_recall_target']=(result['event']['recall']>=args.minimum_recall
                                           and result['minimum_game_recall']>=args.minimum_recall)
            result['iou30']=summarize([row['iou30'] for row in result['folds']])
            result['iou50']=summarize([row['iou50'] for row in result['folds']])
            result['fit_evaluate_seconds']=time.perf_counter()-start
            report['candidates'].append(result)
            # Durable partial results if interrupted; final destination appears only on completion.
            partial=args.out.with_suffix('.partial.json')
            partial.write_text(json.dumps(report,indent=2))
            print(json.dumps({k:result[k] for k in ('id','event','minimum_game_recall','fit_evaluate_seconds')}),flush=True)
    candidates=report['candidates']
    qualified=[r for r in candidates if r['meets_recall_target']]
    best=(max(qualified,key=lambda r:(r['event']['precision'],-r['event']['boundary_p90'],-r['sample_fps']))
          if qualified else max(candidates,key=lambda r:(r['minimum_game_recall'],r['event']['recall'],r['event']['precision'])))
    fastest=min(r['sample_fps'] for r in candidates)
    fast=max([r for r in candidates if r['sample_fps']==fastest],key=lambda r:(r['event']['recall'],r['event']['precision']))
    report['final_holdout']=[]
    choices={} if args.development_only else {r['id']:r for r in (best,fast)}
    for choice in choices.values():
        cfg=configuration(choice['variant'],choice['sample_fps'])
        if 'exclude' in specification:cfg.model_exclude=tuple(specification['exclude'])
        cases=loaded[cfg.sample_fps]
        ds=model.build(cfg,[(r['name'],f,m) for r,f,m in cases if r['partition']=='development'])
        classifier=fit_candidate(cfg,ds,choice['variant']);evaluation=dict(id=choice['id'],games=[])
        for r,f,marks in cases:
            if r['partition']!='holdout':continue
            pred=model.intervals(cfg,f.t,model.predict(classifier,f,cfg))
            evaluation['games'].append(dict(game=r['name'],predictions=pred,
                event=interval_metrics(r['marks'],pred,0),
                iou30=interval_metrics(r['marks'],pred,.3),iou50=interval_metrics(r['marks'],pred,.5)))
        evaluation['event']=summarize([r['event'] for r in evaluation['games']])
        evaluation['iou50']=summarize([r['iou50'] for r in evaluation['games']])
        evaluation['iou30']=summarize([r['iou30'] for r in evaluation['games']])
        export=args.out.parent/(args.out.stem+'-'+choice['id']+'-development-only.json')
        if specification['schema']=='scout-person-v1' and choice['variant']!='linear-balanced':
            if export.exists():raise ValueError('Candidate export already exists')
            model.save(export,classifier,ds.names,cfg)
            evaluation['classifier_file']=export.name
        else:
            evaluation['classifier_file']=None
            evaluation['not_exported']='Experimental feature/model schema; production runtime integration pending'
        report['final_holdout'].append(evaluation)
        print(json.dumps({k:evaluation[k] for k in ('id','event')}),flush=True)
    report['selected_on_development']=best['id']
    with args.out.open('x') as output:json.dump(report,output,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
