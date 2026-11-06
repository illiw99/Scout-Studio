#!/usr/bin/env python3
"""Build a private schema-two pack from an unchanged frozen research candidate.

No training, footage processing, activation or publishing. Inputs are explicit;
the managed graph must attest to the same checkpoint hash and input resolution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.classifier_runtime import Classifier,DYNAMIC_FORMAT,CONTEXT_FORMAT,FEATURE_SETTINGS
from scout.model_packs import build,POSTPROCESS_V2,sha
from scout.inference import REFERENCE_MODELS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head',type=Path,required=True)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--license-file',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='new private workspace directory')
    parser.add_argument('--id',default='scout-reviewed-balanced')
    parser.add_argument('--version',default='0.2.0')
    parser.add_argument('--reference-model',choices=sorted(REFERENCE_MODELS),default='yolo26n.pt')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new pack workspace')
    head=json.loads(args.head.read_text())
    if (head.get('format')!='scout-research-head-v1' or head['pipeline'].get('region_policy')!='dynamic'
            or head['pipeline'].get('tracker_policy')!='time-scaled'):raise ValueError('Expected the frozen dynamic/time-scaled candidate')
    exported=json.loads((args.graph.parent/'manifest.json').read_text());backend='openvino' if args.graph.suffix=='.xml' else 'onnx'
    if (args.graph.suffix not in ('.xml','.onnx') or exported['identity']['source']!=head['weights_sha256']
            or exported['identity']['imgsz']!=head['imgsz'] or exported['identity']['signature']['engine']!=backend):
        raise ValueError('Graph provenance differs from frozen candidate')
    graph_files=[args.graph]+([args.graph.with_suffix('.bin')] if backend=='openvino' else [])
    for file in graph_files:
        if exported['files'].get(file.name)!=sha(file):raise ValueError('Managed graph checksum mismatch')
    settings={key:head['configuration'][key] for key in FEATURE_SETTINGS}
    target_format=CONTEXT_FORMAT if head['classifier']['format']==CONTEXT_FORMAT else DYNAMIC_FORMAT
    core=dict(head['classifier'],format=target_format,feature_pipeline='dynamic-field-v1',feature_settings=settings)
    old,new=Classifier(head['classifier']),Classifier(core)
    probes=np.random.default_rng(0).normal(size=(256,len(core['names'])))
    if not np.array_equal(old.predict_proba(probes),new.predict_proba(probes)):raise ValueError('Classifier changed during packaging')
    args.out.mkdir(parents=True);inputs=args.out/'inputs';inputs.mkdir()
    (inputs/'classifier.json').write_text(json.dumps(core,separators=(',',':')))
    shutil.copyfile(args.license_file,inputs/'LICENSE.txt')
    for file in graph_files:shutil.copyfile(file,inputs/('detector'+file.suffix))
    post={key:head['configuration'].get(key,0.) for key in POSTPROCESS_V2}
    post.update(model_peak_threshold=head['postprocessing'].get('peak_threshold',0.),
        model_refine_fraction=head['postprocessing'].get('refine_fraction',0.),
        model_refine_padding=head['postprocessing'].get('refine_padding',0.))
    manifest=dict(schema=2,id=args.id,version=args.version,backend=backend,runtime_api=2,
        preprocessing='scout-coco-person-resized-v1',classifier=target_format,feature_pipeline='dynamic-field-v1',
        feature_settings=settings,imgsz=head['imgsz'],sample_fps=head['configuration']['sample_fps'],settings=post,
        cpu_threads=8,gpu_batch=16,gpu_reference=dict(name=args.reference_model,sha256=head['weights_sha256']),
        license='Ultralytics detector AGPL-3.0; private reviewed classifier evaluation',
        source='Official Ultralytics v8.4.0 reference asset and a development-only frozen Scout decision head',
        frozen_head_sha256=hashlib.sha256(args.head.read_bytes()).hexdigest(),
        files={file.name:dict(size=file.stat().st_size,sha256=sha(file)) for file in inputs.iterdir()})
    approval=args.out/'approval.json';approval.write_text(json.dumps(manifest,indent=2))
    archive=args.out/(args.id+'-'+args.version+'.zip');digest=build(approval,inputs,archive)
    (args.out/'result.json').write_text(json.dumps(dict(file=archive.name,sha256=digest),indent=2))
    print(json.dumps(dict(file=str(archive),sha256=digest)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
