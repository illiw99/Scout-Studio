#!/usr/bin/env python3
"""Export a trusted, hash-pinned research checkpoint into numeric inference JSON.

Training tool only. Never give this tool a checkpoint from an untrusted source:
joblib uses pickle. The application itself reads only validated numeric JSON.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import joblib
    from scout.classifier_runtime import load,export,BOUNDARY_FORMAT
    from scout.boundary_runtime import FORMAT,feature_names
    from scout.config import Config
    _,document=load(args.classifier)
    if document['format']!='scout-hgb-v3':raise ValueError('Require the frozen base context classifier')
    if args.checkpoint.stat().st_size>16*1024*1024:raise ValueError('Oversized checkpoint')
    payload=args.checkpoint.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=args.checkpoint_sha256:raise ValueError('Checkpoint identity differs')
    heads=joblib.load(io.BytesIO(payload))
    if not isinstance(heads,list) or len(heads)!=2:raise ValueError('Expected start and end heads')
    exported=[]
    names=feature_names(document['names'])
    with tempfile.TemporaryDirectory(prefix='scout-boundary-export-') as directory:
        for index,head in enumerate(heads):
            path=Path(directory)/f'head-{index}.json'
            cfg=Config(model_exclude=())
            export(path,head,names,cfg,cfg.model_offsets)
            raw=json.loads(path.read_text())
            exported.append({key:raw[key] for key in ('format','names','baseline','trees')})
        document['format']=BOUNDARY_FORMAT
        document['boundary_refiner']=dict(format=FORMAT,names=document['names'],heads=exported)
        path=Path(directory)/'combined.json'
        path.write_text(json.dumps(document,allow_nan=False,separators=(',',':'))+'\n')
        load(path)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('xb') as stream:stream.write(path.read_bytes())
    print(f'Exported {args.out.stat().st_size} bytes; checkpoint {args.checkpoint_sha256}')


if __name__=='__main__':
    main()
