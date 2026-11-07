#!/usr/bin/env python3
"""Game-grouped training/tuning for exchangeable temporal play predictors.

Validation arrays are read only after development-only selection is frozen.
No source footage, reviews or production models are written.
"""
import argparse,hashlib,json,sys,time,platform,zipfile
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.config import Config
from scout.play_approaches import sequence_windows,complete_postprocessing,prepare_features,visual_subsample
from scout.temporal_training import build_head,normalization
from scout.model_evaluation import interval_metrics,summarize
from scout import model


def probabilities(net,case,offsets,device):
 import torch
 indices=sequence_windows(case['t'],case['X'],offsets);parts=[];net.eval()
 with torch.inference_mode():
  for start in range(0,len(indices),256):
   batch=torch.from_numpy(case['X'][indices[start:start+256]].astype(np.float32)).to(device)
   parts.append(net(batch).sigmoid().cpu().numpy())
 return np.concatenate(parts)


def fit(cases,variant,device,seed):
 import torch
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mean,scale=normalization(cases)
 net=build_head(cases[0]['X'].shape[1],len(variant['offsets']),architecture=variant['architecture'],width=variant['width'],layers=2,mean=mean,scale=scale).to(device)
 optimizer=torch.optim.AdamW(net.parameters(),lr=variant['lr'],weight_decay=.01)
 indices=[sequence_windows(c['t'],c['X'],variant['offsets']) for c in cases]
 classes=[([np.flatnonzero(c['y']==i) for i in (0,1)]) for c in cases]
 if any(not len(v) for pair in classes for v in pair):raise ValueError('Every training game needs positive and negative frames')
 net.train()
 for epoch in range(variant['epochs']):
  for step in range(24):
   batch=[];labels=[]
   for _ in range(4):
    i=int(rng.integers(len(cases)));klass=int(rng.integers(2));rows=rng.choice(classes[i][klass],64)
    batch.append(cases[i]['X'][indices[i][rows]]);labels.append(cases[i]['y'][rows])
   X=torch.from_numpy(np.concatenate(batch).astype(np.float32)).to(device)
   y=torch.from_numpy(np.concatenate(labels).astype(np.float32)).to(device)
   loss=torch.nn.functional.binary_cross_entropy_with_logits(net(X),y)
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
 return net.eval()


def score(cases,probs,parameters):
 cfg=Config(sample_fps=cases[0]['fps'],**parameters);metrics=[];per_game=[];strict=[]
 for case,prob in zip(cases,probs):
  prediction=model.postprocess(cfg,case['t'],prob)
  end=max(s['start']+s['duration'] for s in case['record']['clips'])
  prediction=[(max(0.,start),min(end,stop)) for start,stop in prediction if min(end,stop)>max(0.,start)]
  result=interval_metrics(case['record']['marks'],prediction,0);metrics.append(result)
  strict.append(interval_metrics(case['record']['marks'],prediction,.5))
  per_game.append(dict(game=case['record']['name'],**summarize([result])))
 return dict(event=summarize(metrics),iou50=summarize(strict),minimum_game_recall=min(g['recall'] for g in per_game),per_game=per_game)


def tune(cases,probs,extended=False):
 trials=[]
 if extended:
  baseline,baseline_trials=tune(cases,probs,False)
  trials.append(baseline)
  for threshold in (.05,.1,.2,.35,.5,.65,.8,.9,.95,.99):
   for low in (.45,.8,.95):
    for gap in (0.,1.5):
     parameters=complete_postprocessing(dict(model_threshold=threshold,model_low_ratio=low,merge_gap=gap,model_smooth=.2,model_min_duration=2.))
     trials.append(dict(parameters=parameters,**score(cases,probs,parameters)))
  def rank(t):return (t['minimum_game_recall']>=.95,t['event']['precision'] if t['minimum_game_recall']>=.95 else t['minimum_game_recall'])
  seeds=sorted(trials,key=rank,reverse=True)[:8]+[baseline]
  for seed in seeds:
   for smooth in (.2,.6):
    for duration in (1.,2.,4.):
     for peak in (0.,.6):
      for refine in (0.,.5):
       parameters=dict(seed['parameters'],model_smooth=smooth,model_min_duration=duration,model_peak_threshold=peak,model_refine_fraction=refine,model_refine_padding=.2)
       trials.append(dict(parameters=parameters,**score(cases,probs,parameters)))
  qualified=[t for t in trials if t['minimum_game_recall']>=.95]
  best=max(qualified,key=lambda t:(t['event']['precision'],-(t['event']['boundary_p90'] or 0))) if qualified else max(trials,key=lambda t:(t['minimum_game_recall'],t['event']['precision']))
  return best,len(trials)+baseline_trials-1
 for threshold in (.05,.1,.2,.35,.5,.65):
  for smooth in (.2,.6):
   for duration in (2.,4.):
    for peak in (0.,.6):
     for refine in (0.,.5):
      parameters=complete_postprocessing(dict(model_threshold=threshold,model_smooth=smooth,model_min_duration=duration,model_peak_threshold=peak,model_refine_fraction=refine,model_refine_padding=.2))
      summary=score(cases,probs,parameters);trials.append(dict(parameters=parameters,**summary))
 qualified=[t for t in trials if t['minimum_game_recall']>=.95]
 best=max(qualified,key=lambda t:(t['event']['precision'],-(t['event']['boundary_p90'] or 0))) if qualified else max(trials,key=lambda t:(t['minimum_game_recall'],t['event']['precision']))
 return best,len(trials)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
 p.add_argument('--source',choices=('person','s3d','mobilenet-v3-small'),required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
 p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=8)
 p.add_argument('--variants',nargs='+',choices=('transformer32','tcn32','transformer64'))
 p.add_argument('--wait-for-features',action='store_true')
 p.add_argument('--feature-transform',choices=('raw','game-zscore'),default='raw')
 p.add_argument('--extended-calibration',action='store_true')
 p.add_argument('--reuse-oof',type=Path)
 p.add_argument('--rate',type=float,help='independent visual-frame/clip sampling; person features cannot be thinned')
 a=p.parse_args()
 if a.out.exists():raise ValueError('Choose a new comparison output')
 import torch
 torch.set_num_threads(4);records=json.loads((a.study/'manifest.json').read_text());folder=a.study/'person' if a.source=='person' else a.study/'visual'/a.source
 base_fps=2.5 if a.source!='s3d' else 1.
 fps=a.rate or base_fps
 if fps<=0 or abs(base_fps/fps-round(base_fps/fps))>1e-8 or fps>base_fps or (a.source=='person' and fps!=base_fps):raise ValueError('Invalid feature sampling comparison')
 def load(record):
  path=folder/(record['name']+'.npz');deadline=time.monotonic()+7200
  while True:
   try:
    with np.load(path,allow_pickle=False) as z:
     t,X=z['t'],z['X'].astype(np.float32);names=z['names'].tolist() if 'names' in z else [a.source+'_'+str(i) for i in range(X.shape[1])]
    break
   except (FileNotFoundError,zipfile.BadZipFile,EOFError,OSError):
    if not a.wait_for_features or time.monotonic()>deadline:raise
    time.sleep(5)
  if fps!=base_fps:t,X=visual_subsample(t,X,record['clips'],base_fps,fps)
  X=prepare_features(X,a.feature_transform)
  y=np.zeros(len(t),np.int64)
  for start,end in record['marks']:y[(t>=start)&(t<end)]=1
  return dict(record=record,t=t,X=X,names=names,y=y,fps=fps)
 dev=[load(r) for r in records if r['partition']=='development']
 if any(c['names']!=dev[0]['names'] or c['X'].shape!=(len(c['t']),len(c['names'])) or not np.isfinite(c['t']).all() or np.any(np.diff(c['t'])<=0) or np.isinf(c['X']).any() for c in dev):raise ValueError('Invalid training feature schema')
 groups=sorted({c['record']['group'] for c in dev})
 report=dict(scope='development tuning followed by repeated held-out-game validation; not a fresh blind test',source=a.source,development_games=[c['record']['name'] for c in dev],validation_games=[r['name'] for r in records if r['partition']!='development'],variants=[])
 report['feature_transform']=a.feature_transform;report['sample_fps']=fps
 report['versions']=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=a.device,seed=17)
 report['training_features_sha256']={c['record']['name']:hashlib.sha256((folder/(c['record']['name']+'.npz')).read_bytes()).hexdigest() for c in dev}
 report['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 a.out.mkdir(parents=True);(a.out/'probabilities').mkdir()
 offsets=[-10.,-6.,-3.,-1.,0.,1.,3.,6.,10.] if a.source=='person' else [-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.]
 variants=[dict(id='transformer32',architecture='transformer',width=32,epochs=a.epochs,lr=.001,offsets=offsets),
           dict(id='tcn32',architecture='tcn',width=32,epochs=a.epochs,lr=.001,offsets=offsets),
           dict(id='transformer64',architecture='transformer',width=64,epochs=a.epochs+4,lr=.0005,offsets=offsets)]
 if a.variants:variants=[v for v in variants if v['id'] in a.variants]
 for variant in variants:
  started=time.perf_counter();oof={}
  if a.reuse_oof:
   previous=json.loads((a.reuse_oof/'selection.json').read_text())
   old_variant=next(r['variant'] for r in previous['variants'] if r['variant']['id']==variant['id'])
   if (previous.get('feature_transform','raw')!=a.feature_transform or old_variant!=variant
       or previous['training_features_sha256']!=report['training_features_sha256']):raise ValueError('OOF reuse identity differs')
   for case in dev:
    with np.load(a.reuse_oof/'probabilities'/(variant['id']+'-'+case['record']['name']+'.npz'),allow_pickle=False) as z:
     if not np.array_equal(z['t'],case['t']):raise ValueError('OOF clock differs')
     oof[case['record']['name']]=z['prob']
  else:
   for group in groups:
    training=[c for c in dev if c['record']['group']!=group];validation=[c for c in dev if c['record']['group']==group]
    net=fit(training,variant,a.device,17)
    for case in validation:oof[case['record']['name']]=probabilities(net,case,variant['offsets'],a.device)
    del net
  probs=[oof[c['record']['name']] for c in dev];best,trials=tune(dev,probs,a.extended_calibration)
  row=dict(variant=variant,development=best,calibration_trials=trials,training_calibration_seconds=time.perf_counter()-started)
  report['variants'].append(row)
  for case,prob in zip(dev,probs):np.savez_compressed(a.out/'probabilities'/(variant['id']+'-'+case['record']['name']+'.npz'),t=case['t'],prob=prob)
  (a.out/'development.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(source=a.source,id=variant['id'],development=best['event'],minimum_game_recall=best['minimum_game_recall'])),flush=True)
 qualified=[r for r in report['variants'] if r['development']['minimum_game_recall']>=.95]
 selected=max(qualified,key=lambda r:r['development']['event']['precision']) if qualified else max(report['variants'],key=lambda r:(r['development']['minimum_game_recall'],r['development']['event']['precision']))
 report['selection']=selected['variant']['id'];report['selection_frozen_before_validation']=True
 (a.out/'selection.json').write_text(json.dumps(report,indent=2))
 net=fit(dev,selected['variant'],a.device,17).cpu();variant=selected['variant']
 torch.save(net.state_dict(),a.out/'head-state.pt')
 class Probability(torch.nn.Module):
  def __init__(self,head):super().__init__();self.head=head
  def forward(self,X):return self.head(X).sigmoid()
 wrapper=Probability(net).eval();dummy=torch.zeros(2,len(offsets),dev[0]['X'].shape[1])
 torch.onnx.export(wrapper,(dummy,),str(a.out/'head.onnx'),input_names=['observations'],output_names=['probabilities'],dynamic_axes={'observations':{0:'batch'},'probabilities':{0:'batch'}},opset_version=17,dynamo=False)
 # Holdout features and labels are first loaded here, after all settings are frozen.
 heldout=[load(r) for r in records if r['partition']!='development'];net=net.to(a.device)
 import onnxruntime as ort
 options=ort.SessionOptions();options.intra_op_num_threads=4
 session=ort.InferenceSession(str(a.out/'head.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
 probs=[];errors=[]
 for c in heldout:
  if c['names']!=dev[0]['names']:raise ValueError('Validation feature schema differs')
  indices=sequence_windows(c['t'],c['X'],offsets);parts=[]
  for first in range(0,len(indices),256):parts.append(session.run(None,{'observations':c['X'][indices[first:first+256]].astype(np.float32)})[0])
  prob=np.concatenate(parts);probs.append(prob);errors.append(float(np.max(np.abs(prob-probabilities(net,c,offsets,a.device)))))
 report['maximum_onnx_reference_probability_error']=max(errors)
 if max(errors)>1e-5:raise ValueError('Exported head differs from training runtime')
 report['validation']=score(heldout,probs,selected['development']['parameters'])
 for c,prob in zip(heldout,probs):np.savez_compressed(a.out/'probabilities'/('validation-'+c['record']['name']+'.npz'),t=c['t'],prob=prob)
 report['names']=dev[0]['names'];report['sample_fps']=fps
 report['asset_sha256']=hashlib.sha256((a.out/'head.onnx').read_bytes()).hexdigest()
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(source=a.source,selected=report['selection'],validation=report['validation'])),flush=True)

if __name__=='__main__':main()
