"""Isolated, bounded real-source timing through the complete pipeline interface."""
from __future__ import annotations
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from . import media,detect,shotcut
from .play_approaches import create_pipeline


def benchmark(cfg, clips, output, *, index=0, offset=0., seconds=180., requested_device='cpu',setup_seconds=0.):
    cfg=replace(cfg)
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('Choose a new benchmark output directory')
    output=output.resolve()
    if not clips or not 0<=index<len(clips):raise ValueError('Invalid benchmark source index')
    source=clips[index]
    if not math.isfinite(seconds) or not math.isfinite(offset) or seconds<=0 or offset<0 or offset+seconds>source.info.duration+.001:raise ValueError('Benchmark window outside source')
    before=source.path.stat()
    cfg.cache_dir=str(output/'detections');cfg.out_dir=str(output/'output')
    if any(cfg.path(path).resolve()!=output/name for path,name in
           ((cfg.cache_dir,'detections'),(cfg.out_dir,'output'))):
        raise ValueError('Benchmark output cannot use a redirected game-state path')
    output.mkdir(parents=True)
    clip=replace(source,start=source.start+offset,info=replace(source.info,duration=seconds))
    original_media=media.iter_frames;original_detector=detect.iter_frames
    def bounded(path,*args,**kwargs):
        if (Path(path).resolve()==source.path.resolve() and len(args)<=3
                and 'start' not in kwargs and 'duration' not in kwargs):
            kwargs.update(start=offset,duration=seconds)
        return original_media(path,*args,**kwargs)
    # This function is deliberately process-local (CLI/worker benchmark only),
    # never called in the Qt UI process or alongside another media pipeline.
    started=time.perf_counter()
    media.iter_frames=bounded;detect.iter_frames=bounded
    try:
        pipeline=create_pipeline(cfg.play_pipeline)
        result=pipeline.predict([clip],cfg)
        review=output/'review.mlt'
        shotcut.write_review(cfg,clips,result.plays,review)
        elapsed=time.perf_counter()-started+setup_seconds
    finally:
        media.iter_frames=original_media;detect.iter_frames=original_detector
    after=source.path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during benchmark')
    report=dict(scope='bounded real-source pipeline including configuration/model initialization/decode/prediction/review writing; excludes interpreter imports and MP4 export encoding',
        pipeline=cfg.play_pipeline or 'established-person-pipeline',pipeline_sha256=cfg.play_pipeline_sha256,
        requested_device=requested_device,effective_device=result.timings.get('device',cfg.device),
        detector_backend=cfg.detector_backend,inference_threads=cfg.inference_threads,
        source=source.path.name,source_offset=offset,video_seconds=seconds,seconds=elapsed,realtime=seconds/elapsed,
        frames=len(result.scores.t) if result.scores is not None else None,plays=len(result.plays),timings=result.timings,
        setup_seconds=setup_seconds,output=str(output),original_source_unchanged=True)
    (output/'benchmark.json').write_text(json.dumps(report,indent=2))
    return report
