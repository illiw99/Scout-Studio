#!/usr/bin/env python3
"""Private development features without person detection: optical flow and S3D.

Reads only sources named by a frozen reviewed-game snapshot. All outputs go to a
new external directory. Reserved games are never decoded by this experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scout import groundtruth,media


MOTION_NAMES=['camera_x','camera_y','camera_speed']
for region in ('field','broad'):
    MOTION_NAMES += [region+'_'+name for name in (
        'p50','p90','p95','p99','p995','above002','above005','above01','above02','above04',
        'mean_x','mean_y','std_x','std_y','coherence','difference','nongreen')]


def motion_vector(rgb,mask,previous,dt):
    if not np.isfinite(dt) or dt<=0 or rgb.shape[:2]!=mask.shape:
        raise ValueError('Motion input geometry and positive time step are required')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if previous is None:return np.zeros(len(MOTION_NAMES),np.float32),gray
    flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,2,5,1.1,0)
    background=~mask
    if background.sum()<100:background=np.ones(mask.shape,bool)
    camera=np.median(flow[background],axis=0)
    residual=(flow-camera)/max(1,gray.shape[0])/dt
    magnitude=np.linalg.norm(residual,axis=2)
    shifted=cv2.warpAffine(previous,np.array([[1,0,camera[0]],[0,1,camera[1]]],np.float32),
                           (gray.shape[1],gray.shape[0]),borderMode=cv2.BORDER_REPLICATE)
    difference=np.abs(gray.astype(np.float32)-shifted.astype(np.float32))/255
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    nongreen=~((hsv[:,:,0]>=28)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=40)&(hsv[:,:,2]>=35))
    broad=np.ones(mask.shape,bool);broad[:int(.2*len(broad))]=False
    values=[float(camera[0]/len(gray)/dt),float(camera[1]/len(gray)/dt),float(np.linalg.norm(camera)/len(gray)/dt)]
    for area in (mask,broad):
        if not area.any():values.extend([0.]*17);continue
        speed=magnitude[area];vectors=residual[area];mean=vectors.mean(0)
        values.extend(np.percentile(speed,[50,90,95,99,99.5]).tolist())
        values.extend([float((speed>threshold).mean()) for threshold in (.002,.005,.01,.02,.04)])
        values.extend([float(mean[0]),float(mean[1]),float(vectors[:,0].std()),float(vectors[:,1].std()),
                       float(np.linalg.norm(mean)/(speed.mean()+1e-9)),float(difference[area].mean()),float(nongreen[area].mean())])
    return np.asarray(values,np.float32),gray


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--size',type=int,choices=(112,224),default=112)
    parser.add_argument('--batch',type=int,default=8)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a new experiment directory')
    if args.batch<=0:raise ValueError('Batch must be positive')
    records=json.loads((args.snapshot/'dataset.json').read_text())
    args.out.mkdir(parents=True)
    import torch
    from torchvision.models.video import s3d,S3D_Weights
    torch.set_num_threads(8);torch.hub.set_dir(str(args.snapshot/'torch-hub'))
    network=s3d(weights=S3D_Weights.DEFAULT).eval()
    encoder=torch.nn.Sequential(network.features,torch.nn.AdaptiveAvgPool3d(1),network.classifier,torch.nn.Flatten(1)).to(args.device).eval()
    if args.device=='cuda':encoder=encoder.half()
    mean=torch.tensor([.43216,.394666,.37645],device=args.device).view(1,3,1,1,1)
    std=torch.tensor([.22803,.22145,.216989],device=args.device).view(1,3,1,1,1)
    for name,specification in (
        ('motion',dict(schema='scout-motion-v1',names=MOTION_NAMES,rates=[5.],exclude=[])),
        ('s3d',dict(schema='scout-s3d-logits-v1',names=[f'kinetics_logit_{i:03d}' for i in range(400)],rates=[1.],exclude=[]))):
        folder=args.out/name;folder.mkdir();(folder/'dataset.json').write_text(json.dumps(records,indent=2))
        (folder/'features.json').write_text(json.dumps(specification,indent=2))
    progress=[]
    for record in records:
        if record['partition']!='development':continue
        game=record['name'];motion_parts=[];video_parts=[];motion_times=[];video_times=[]
        clips=groundtruth.project_clips(args.snapshot/game/'review.mlt')
        for clip,identity in zip(clips,record['clips']):
            before=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(identity['bytes'],identity['mtime_ns']):raise ValueError('Source changed since snapshot')
            original_mask=np.load(args.snapshot/game/(clip.path.stem+'-mask.npy'),allow_pickle=False)
            mask=cv2.resize(original_mask.astype(np.uint8),(320,180),interpolation=cv2.INTER_NEAREST).astype(bool)
            previous=None;motion=[];frames=[];start=time.perf_counter()
            for rgb in media.iter_frames(clip.path,5,320,180):
                row,previous=motion_vector(rgb,mask,previous,.2);motion.append(row)
                frames.append(cv2.resize(rgb,(args.size,args.size),interpolation=cv2.INTER_AREA))
            decode_motion_seconds=time.perf_counter()-start
            if not frames:raise ValueError('No decoded frames')
            motion_parts.append(np.asarray(motion));motion_times.append(clip.start+np.arange(len(frames))/5)
            centers=np.arange(0,len(frames),5);logits=[];start=time.perf_counter()
            with torch.inference_mode():
                for first in range(0,len(centers),args.batch):
                    selected=centers[first:first+args.batch]
                    batch=np.stack([np.stack([frames[i] for i in np.clip(np.arange(center-7,center+9),0,len(frames)-1)]) for center in selected])
                    tensor=torch.from_numpy(batch).to(args.device).permute(0,4,1,2,3).float()/255
                    tensor=(tensor-mean)/std
                    if args.device=='cuda':tensor=tensor.half()
                    logits.append(encoder(tensor).float().cpu().numpy())
            encoder_seconds=time.perf_counter()-start
            video_parts.append(np.concatenate(logits));video_times.append(clip.start+centers/5)
            after=clip.path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during extraction')
            result=dict(game=game,clip=clip.path.name,video_seconds=clip.info.duration,decoded_frames=len(frames),
                        decode_and_motion_seconds=decode_motion_seconds,s3d_encoder_seconds=encoder_seconds,
                        device=args.device,size=args.size,scope='shared extraction run; not isolated throughput benchmark')
            progress.append(result);(args.out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(result),flush=True)
        for name,parts,times,fps,names in [('motion',motion_parts,motion_times,5,MOTION_NAMES),('s3d',video_parts,video_times,1,[f'kinetics_logit_{i:03d}' for i in range(400)])]:
            folder=args.out/name/game;folder.mkdir()
            np.savez_compressed(folder/f'features-{fps}.npz',t=np.concatenate(times),X=np.concatenate(parts),names=np.asarray(names))
        print(game,'motion and direct-video features ready',flush=True)
    metadata=dict(model='TorchVision S3D Kinetics400',weights_url=S3D_Weights.DEFAULT.url,
        weights_sha256=hashlib.sha256((args.snapshot/'torch-hub/checkpoints/s3d-d76dad2f.pth').read_bytes()).hexdigest(),
        frame_fps=5,video_feature_fps=1,frames_per_clip=16,context='7 past and8 future frames; replicated at source edges',
        preprocessing=f'full-frame resize{args.size}; official RGB mean/std; adaptive pooling',torch=torch.__version__)
    (args.out/'provenance.json').write_text(json.dumps(metadata,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
