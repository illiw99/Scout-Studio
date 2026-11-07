#!/usr/bin/env python3
"""Read a private frozen source manifest; write reusable independent encoder features."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout.video_encoders import VisualEncoder,encode_source_many
from scout.storage import validate_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mobile-weights',type=Path,required=True);p.add_argument('--s3d-weights',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--batch',type=int,default=32)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a new extraction directory')
    records=json.loads(a.manifest.read_text());a.out.mkdir(parents=True)
    encoders={k:VisualEncoder(k,path,device=a.device) for k,path in
              [('mobilenet-v3-small',a.mobile_weights),('s3d',a.s3d_weights)]}
    progress=[]
    for record in records:
        validate_name(record['name']);parts={k:[] for k in encoders};times={k:[] for k in encoders}
        for source in record['clips']:
            path=Path(source['path']);before=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(source['bytes'],source['mtime_ns']):raise ValueError('Source changed')
            encoded=encode_source_many(path,source['start'],encoders,device=a.device,batch=a.batch)
            row=dict(game=record['name'],source=path.name,video_seconds=source['duration'],encoders={})
            for kind,(t,X,timing) in encoded.items():
                parts[kind].append(X);times[kind].append(t)
                row['encoders'][kind]=dict(**timing,frames=len(t))
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed while decoding')
            progress.append(row);(a.out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(row),flush=True)
        for kind in encoders:
            folder=a.out/kind;folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/(record['name']+'.npz'),t=np.concatenate(times[kind]),X=np.concatenate(parts[kind]).astype(np.float16))
    (a.out/'manifest.json').write_text(a.manifest.read_text())
    (a.out/'provenance.json').write_text(json.dumps(dict(device=a.device,decode_fps=5,
        assets={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in [('mobile',a.mobile_weights),('s3d',a.s3d_weights)]},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
