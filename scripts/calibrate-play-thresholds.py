#!/usr/bin/env python3
"""Calibrate on development out-of-fold probabilities; never read held-out games."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.config import Config
from scout.storage import validate_name
from scout.model import intervals
from scout.model_evaluation import interval_metrics,summarize
from itertools import product


from scout.model import filter_by_peak,refine_boundaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--minimum-recall',type=float,default=.95)
    parser.add_argument('--minimum-durations',type=float,nargs='+',default=[1.5,2.])
    parser.add_argument('--peak-thresholds',type=float,nargs='+',default=[0.],help='optional proposal peak-probability filter; experimental postprocessing')
    parser.add_argument('--candidates',nargs='+',help='evaluate only these candidate IDs')
    parser.add_argument('--refine-fractions',type=float,nargs='+',default=[0.],help='crop proposals to a fraction of their own peak; zero disables refinement')
    parser.add_argument('--refine-padding',type=float,default=0.)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new calibration output')
    if not 0<args.minimum_recall<=1:raise ValueError('Invalid recall target')
    if any(not np.isfinite(x) or x<=0 for x in args.minimum_durations):raise ValueError('Invalid minimum duration')
    if any(not np.isfinite(x) or not 0<=x<=1 for x in args.peak_thresholds):raise ValueError('Invalid peak threshold')
    if any(not np.isfinite(x) or not 0<=x<=1 for x in args.refine_fractions) or not np.isfinite(args.refine_padding) or args.refine_padding<0:raise ValueError('Invalid boundary refinement')
    report=json.loads(args.comparison.read_text());expected=set(report['development_games'])
    if args.candidates and not set(args.candidates)<={c['id'] for c in report['candidates']}:raise ValueError('Unknown candidate')
    if expected & set(report['holdout_games']):raise ValueError('Training/holdout overlap')
    probability_directory=report.get('probability_directory','probabilities')
    validate_name(probability_directory)
    output=dict(schema=1,scope='development calibration/selection scores, not unbiased final accuracy',
                minimum_game_recall=args.minimum_recall,holdout_used=False,candidates=[])
    for candidate in report['candidates']:
        if args.candidates and candidate['id'] not in args.candidates:continue
        validate_name(candidate['id'])
        folder=args.comparison.parent/probability_directory/candidate['id']
        files=list(folder.glob('*.npz'))
        if {p.stem for p in files}!=expected:raise ValueError('Probability files must contain exactly the development games')
        cfg=Config(**candidate['configuration']);cases=[]
        for file in sorted(files):
            with np.load(file,allow_pickle=False) as z:t,prob,truth=z['t'],z['prob'],z['truth']
            if (t.shape!=prob.shape or t.ndim!=1 or not len(t) or np.any(np.diff(t)<=0)
                    or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1))):raise ValueError('Invalid probability snapshot')
            cases.append((file.stem,t,prob,truth))
        trials=[];seen=set()
        for threshold in (.1,.15,.2,.25,.3,.4,.5,.6):
            for smooth in (.2,.6):
                for duration,peak,refine in product(args.minimum_durations,args.peak_thresholds,args.refine_fractions):
                    key=(threshold,max(1,round(smooth*cfg.sample_fps)),duration,peak,refine)
                    if key in seen:continue
                    seen.add(key)
                    trial=replace(cfg,model_threshold=threshold,model_smooth=smooth,model_min_duration=duration)
                    metrics=[];per_game=[]
                    for name,t,prob,truth in cases:
                        predictions=filter_by_peak(t,prob,intervals(trial,t,prob),peak)
                        predictions=refine_boundaries(t,prob,predictions,refine,args.refine_padding)
                        m=interval_metrics(truth,predictions,0);metrics.append(m)
                        per_game.append(dict(game=name,**summarize([m])))
                    summary=summarize(metrics)
                    minimum=min(g['recall'] for g in per_game)
                    trials.append(dict(threshold=threshold,smooth=smooth,minimum_duration=duration,peak_threshold=peak,
                        refine_fraction=refine,refine_padding=args.refine_padding,
                        event=summary,minimum_game_recall=minimum,meets_target=minimum>=args.minimum_recall,
                        per_game=per_game))
        qualified=[t for t in trials if t['meets_target']]
        best=(max(qualified,key=lambda t:(t['event']['precision'],-t['event']['boundary_p90'])) if qualified else
              max(trials,key=lambda t:(t['minimum_game_recall'],t['event']['recall'],t['event']['precision'])))
        result=dict(id=candidate['id'],sample_fps=cfg.sample_fps,configuration=candidate['configuration'],best=best,trials=trials)
        output['candidates'].append(result)
        print(json.dumps(dict(id=result['id'],threshold=best['threshold'],event=best['event'],minimum_game_recall=best['minimum_game_recall'],meets_target=best['meets_target'])),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(output,stream,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
