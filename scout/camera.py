"""Camera geometry for play features; never edits saved field polygons."""
from __future__ import annotations
import numpy as np


def project_field_region(reference, current, polygon, width=640):
    """Experimental camera projection; return None instead of guessing on failure.

    Does not edit a polygon or any file. Matching is independent of play labels.
    Callers opt in through a versioned feature pipeline.
    """
    import cv2
    points=np.asarray(polygon,dtype=np.float32)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all()
            or reference.ndim!=3 or current.ndim!=3 or width<64):
        raise ValueError('Invalid field projection inputs')
    def small(image):
        h,w=image.shape[:2];sw=min(width,w);sh=max(1,round(h*sw/w))
        gray=cv2.cvtColor(cv2.resize(image,(sw,sh)),cv2.COLOR_RGB2GRAY)
        return gray,np.diag([sw/w,sh/h,1.])
    first,scale_ref=small(reference);second,scale_current=small(current)
    detector=cv2.ORB_create(nfeatures=2500)
    k1,d1=detector.detectAndCompute(first,None);k2,d2=detector.detectAndCompute(second,None)
    quality=dict(matches=0,inliers=0,inlier_fraction=0.,accepted=False)
    if d1 is None or d2 is None:return None,quality
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
    quality['matches']=len(good)
    if len(good)<20:return None,quality
    source=np.float32([k1[m.queryIdx].pt for m in good]);target=np.float32([k2[m.trainIdx].pt for m in good])
    matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,3.)
    if matrix is None or inliers is None:return None,quality
    accepted=inliers[:,0].astype(bool);count=int(accepted.sum());fraction=count/len(good)
    quality.update(inliers=count,inlier_fraction=fraction)
    if count<20 or fraction<.5:return None,quality
    spread=np.ptp(source[accepted],axis=0)
    if spread[0]<.3*first.shape[1] or spread[1]<.15*first.shape[0]:return None,quality
    matrix=np.linalg.inv(scale_current)@matrix@scale_ref
    source_pixels=points*[reference.shape[1],reference.shape[0]]
    projected=cv2.perspectiveTransform(source_pixels.astype(np.float32)[None],matrix)[0]
    projected=projected/[current.shape[1],current.shape[0]]
    if not np.isfinite(projected).all() or np.abs(projected).max()>4:return None,quality
    original_area=cv2.contourArea(points,oriented=True)
    projected_area=cv2.contourArea(projected.astype(np.float32),oriented=True)
    if original_area==0 or not .25<=projected_area/original_area<=4:return None,quality
    quality['accepted']=True
    return projected.tolist(),quality


def filter_detections(det,keyframes,fallback,cfg):
    """Filter source-coordinate boxes with an interpolated, time-local polygon."""
    from . import field,features
    from .detect import Detections
    if fallback.shape!=det.mask_shape:raise ValueError('Camera fallback geometry differs from detections')
    accepted=sorted((k['at'],k['polygon']) for k in keyframes if k.get('polygon') is not None)
    times=np.asarray([k[0] for k in accepted]);points=np.asarray([k[1] for k in accepted])
    boxes=[];conf=[];offsets=[0]
    for i,when in enumerate(det.times):
        if len(accepted):
            coordinates=np.asarray([np.interp(when,times,points[:,vertex,axis])
                for vertex in range(points.shape[1]) for axis in range(2)]).reshape(-1,2)
            mask=field.polygon_mask(coordinates,fallback.shape,cfg.field_shrink_top)
        else:mask=fallback
        b,c=det.frame(i);keep=features._on_field(b,mask,cfg)
        boxes.append(b[keep]);conf.append(c[keep]);offsets.append(offsets[-1]+int(keep.sum()))
    return Detections(det.times.copy(),np.asarray(offsets,np.int64),
        np.concatenate(boxes) if boxes else np.empty((0,4),np.float32),
        np.concatenate(conf) if conf else np.empty(0,np.float32),det.mask_shape)
