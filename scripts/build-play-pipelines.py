#!/usr/bin/env python3
"""Build complete private pipeline recipes from development-selected comparisons.

Only inference assets are published; GT/features/training states stay private.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.play_approaches import complete_postprocessing,read_manifest
from scout.video_encoders import torch_encoder,VisualEncoder


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_encoders(study,shared):
    import torch
    from scout.video_encoders import ENCODERS
    shared.mkdir(parents=True,exist_ok=True)
    setup=json.loads((study/'setup.json').read_text());provenance=json.loads((study/'visual/provenance.json').read_text())
    paths={'mobilenet-v3-small':Path(setup['mobile_weights']),'s3d':Path(setup['s3d_weights'])}
    report={}
    for kind,path in paths.items():
        expected=provenance['assets']['mobile' if kind=='mobilenet-v3-small' else 's3d']
        if sha(path)!=expected:raise ValueError('Encoder weights changed since extraction')
        reference=shared/(kind+'.pth');cpu=shared/(kind+'.onnx')
        if reference.exists() or cpu.exists():raise ValueError('Encoder exports are immutable; choose a new asset directory')
        shutil.copyfile(path,reference)
        net=torch_encoder(kind,reference).eval();spec=ENCODERS[kind]
        shape=(1,spec['size'],spec['size'],3) if spec['frames']==1 else (1,spec['frames'],spec['size'],spec['size'],3)
        dummy=torch.zeros(shape,dtype=torch.uint8)
        torch.onnx.export(net,(dummy,),str(cpu),input_names=['rgb'],output_names=['features'],dynamic_axes={'rgb':{0:'batch'},'features':{0:'batch'}},opset_version=17,dynamo=False)
        # Numeric parity with real-valued uint8 RGB batches; no detector quality claim.
        rgb=np.random.default_rng(17).integers(0,256,size=shape,dtype=np.uint8)
        with torch.inference_mode():expected_values=net(torch.from_numpy(rgb)).numpy()
        actual=VisualEncoder(kind,cpu,device='cpu',sha256=sha(cpu)).predict(rgb)
        error=float(np.max(np.abs(expected_values-actual)))
        if error>1e-3:raise ValueError('Encoder ONNX parity failed')
        report[kind]=dict(reference={'file':reference.name,'sha256':sha(reference)},cpu={'file':cpu.name,'sha256':sha(cpu)},maximum_cpu_onnx_error=error)
        print(json.dumps(dict(encoder=kind,maximum_cpu_onnx_error=error)),flush=True)
    (shared/'encoders.json').write_text(json.dumps(report,indent=2));return report


def choose(reports):
    qualified=[r for r in reports if next(v for v in r[1]['variants'] if v['variant']['id']==r[1]['selection'])['development']['minimum_game_recall']>=.95]
    def best(r):return next(v for v in r[1]['variants'] if v['variant']['id']==r[1]['selection'])['development']
    return (max(qualified,key=lambda r:best(r)['event']['precision']) if qualified else
            max(reports,key=lambda r:(best(r)['minimum_game_recall'],best(r)['event']['precision'])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--comparisons',type=Path,nargs='*');p.add_argument('--assets',type=Path,help='existing verified encoder asset directory')
    p.add_argument('--encoders-only',action='store_true');a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new immutable output directory')
    a.out.mkdir(parents=True)
    shared=a.assets or a.out/'shared'
    encoders=json.loads((shared/'encoders.json').read_text()) if a.assets else export_encoders(a.study,shared)
    if a.encoders_only:return
    grouped={}
    for directory in a.comparisons or []:
        report=json.loads((directory/'report.json').read_text())
        if not report['selection_frozen_before_validation']:raise ValueError('Selection must be frozen before validation')
        grouped.setdefault(report['source'],[]).append((directory,report))
    rows=[]
    for source,reports in grouped.items():
        directory,report=choose(reports);selected=next(v for v in report['variants'] if v['variant']['id']==report['selection'])
        variant=selected['variant'];transform=report.get('feature_transform','raw');rate=report['sample_fps']
        identity=(source+'-'+transform+'-'+variant['id']+'-'+str(rate).replace('.','p')+'hz').lower()
        folder=a.out/identity;folder.mkdir();head=folder/'head.onnx';shutil.copyfile(directory/'head.onnx',head)
        assets={'head':{'file':head.name,'sha256':sha(head)}}
        if source!='person':
            for key,asset in [('encoder_cpu',encoders[source]['cpu']),('encoder_reference',encoders[source]['reference'])]:
                origin=shared/asset['file']
                if sha(origin)!=asset['sha256']:raise ValueError('Encoder asset changed')
                (folder/asset['file']).hardlink_to(origin)
                assets[key]=asset
        quality=report['validation'];event=quality['event']
        label=('Person' if source=='person' else 'S3D' if source=='s3d' else 'MobileNet')+' / '+variant['id']
        spec=dict(format='scout-play-pipeline-v1',id=identity,label=label,
            approach={'person':'person-temporal','s3d':'s3d-video','mobilenet-v3-small':'mobile-temporal'}[source],
            names=report['names'],offsets=variant['offsets'],sample_fps=rate,assets=assets,
            postprocessing=complete_postprocessing(selected['development']['parameters']),feature_dtype='float32' if source=='person' else 'float16',
            feature_transform=transform,encoder_revision=1,
            evaluation_note=f"Repeated validation on four games; GPU-derived features / CPU ONNX head. Precision {event['precision']:.1%}; recall {event['recall']:.1%}; minimum-game recall {quality['minimum_game_recall']:.1%}. Experimental; default remains the accepted pipeline.",
            quality=dict(precision=event['precision'],recall=event['recall'],minimum_game_recall=quality['minimum_game_recall']),
            training=dict(games=report['development_games'],source_report_sha256=sha(directory/'report.json'),selection_rule='development precision subject to minimum game recall >=95%; no validation selection'))
        if source=='person':spec['requires_model_pack']='scout-reviewed-precision@0.3.0'
        (folder/'pipeline.json').write_text(json.dumps(spec,indent=2));read_manifest(folder/'pipeline.json',verify=True)
        rows.append(dict(source=source,comparison=str(directory),pipeline=str(folder/'pipeline.json'),development=selected['development'],validation=quality))
    (a.out/'catalog.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
