"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .storage import read_library, validate_name, STATE_DIR_NAMES
from . import (timeline, detect, features, segment, cut, report, field, pick,
               shotcut, groundtruth, evaluate, tune, collect,
               merge, model, camera)
from .media import grab_frame

CONFIG_FILE = "scout.json"


def cmd_model_pack(args):
    from .model_packs import ModelStore, build
    store = ModelStore(args.store)
    required = {'build': ('approval', 'inputs', 'out'), 'import': ('file',), 'activate': ('pack',)}
    for name in required.get(args.action, ()):
        if not getattr(args, name):
            raise ValueError(f'--{name} is required for model-pack {args.action}')
    if args.action == 'build':
        result = build(args.approval, args.inputs, args.out)
    elif args.action == 'import':
        result = store.install(args.file, args.sha256)
    elif args.action == 'activate':
        result = store.activate(args.pack)
    elif args.action == 'rollback':
        result = store.rollback()
    else:
        result = '\n'.join(store.list())
    print(result)
    return 0


def cmd_export_classifier(args):
    cfg = _load(args)
    source, destination = cfg.path(args.input), cfg.path(args.out)
    if source.suffix.lower() != '.pkl' or destination.suffix.lower() != '.json':
        raise ValueError('Export a trusted .pkl classifier to a new .json file')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Classifier export destination already exists; choose a new file')
    classifier,names = model.load(source,cfg)
    model.save(destination,classifier,names,cfg)
    print(f'  exported numeric classifier -> {destination}')
    return 0


def cmd_export_detector(args):
    import shutil
    import tempfile
    from .inference import prepare_graph
    cfg = _load(args)
    cfg.device = 'cpu'
    destination = cfg.path(args.out)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Detector export destination already exists; choose a new directory')
    graph = prepare_graph(cfg)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.detector-export-',dir=destination.parent) as directory:
        stage=Path(directory)/'publish';stage.mkdir()
        shutil.copyfile(graph,stage/('detector.onnx' if cfg.detector_backend=='onnx' else 'detector.xml'))
        if cfg.detector_backend=='openvino':
            shutil.copyfile(graph.with_suffix('.bin'),stage/'detector.bin')
        stage.rename(destination)
    print(f'  exported CPU detector -> {destination}')
    return 0


def _load(args) -> Config:
    cfg = Config.load(args.config)
    saved_match = cfg.match
    if saved_match is not None:
        validate_name(saved_match)
    saved_states = {cfg.game_dir / name for name in STATE_DIR_NAMES}
    workspace_roots = {Config(media_dir=cfg.media_dir).root}
    if args.config:
        workspace_roots.add(Path(args.config).resolve().parent)
    # Keep workspace model files usable when selecting an external source root.
    model_roots = [cfg.root]
    if getattr(args, "media_dir", None) is not None:
        model_roots.insert(0, Config(media_dir=args.media_dir).root)
    if args.config:
        model_roots.insert(0, Path(args.config).resolve().parent)
    models = {}
    for key in ("model_path", "model"):
        for root in model_roots:
            candidate = root / getattr(cfg, key)
            if candidate.is_file():
                models[key] = str(candidate.resolve())
                break
    for key in ("media_dir", "source_dir", "out_dir", "sample_fps", "model", "cut_mode",
                "imgsz", "segmenter", "detector_backend", "device"):
        v = getattr(args, key, None)
        if v is not None:
            setattr(cfg, key, v)
    match = getattr(args, "match", None)
    if match:
        validate_name(match)
        cfg.match = match
        if getattr(args, "source_dir", None) is None:
            # A selected name must never retain an unrelated saved source.
            cfg.source_dir = None
            roots = []
            if args.config:
                roots.append(Path(args.config).resolve().parent)
            roots.append(cfg.root)
            for root in dict.fromkeys(roots):
                folder = read_library(root)["games"].get(match)
                if folder:
                    cfg.source_dir = folder
                    break
    # Conventional workspace paths from older saved configs adopt game-local
    # storage. Explicit CLI output and genuinely custom paths stay authoritative.
    conventional = {
        "out_dir": {Path("out"), *saved_states, *(Path(name) for name in STATE_DIR_NAMES)},
        "fields_dir": {Path("fields"), *(state / 'fields' for state in saved_states),
                       *(Path(name) / 'fields' for name in STATE_DIR_NAMES)},
        "cache_dir": {Path(".scout-cache"), *(state / 'cache' for state in saved_states),
                      *(Path(name) / 'cache' for name in STATE_DIR_NAMES)},
    }
    for name in (saved_match, cfg.match):
        if name:
            conventional["out_dir"].add(Path("out") / name)
    for key, values in conventional.items():
        legacy = values | {root / value for root in workspace_roots for value in values}
        if Path(getattr(cfg, key)) in legacy and not (key == "out_dir" and getattr(args, "out_dir", None) is not None):
            setattr(cfg, key, getattr(Config(), key))
    if cfg.source_dir is not None:
        for key, candidate in models.items():
            if getattr(args, key, None) is None:
                setattr(cfg, key, candidate)
        if not cfg.path(cfg.model_path).exists():
            bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
            candidate = bundled / "models" / "play.pkl"
            if candidate.is_file():
                cfg.model_path = str(candidate)
    selection = getattr(args, 'model_pack', None) or cfg.model_pack
    if selection and args.cmd in ('run', 'detect', 'segment', 'preview'):
        from .model_packs import ModelStore
        store = ModelStore(getattr(args, 'model_packs_dir', None) or cfg.model_packs_dir)
        if selection == 'active':
            from .model_defaults import ensure_default
            ensure_default(store)
        store.apply(cfg, selection)
        args.model_pack = cfg.model_pack  # pin active selection across run's repeated loads
    for name in ("sample_fps", "imgsz", "batch", "clip_workers"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if type(cfg.inference_threads) is not int or cfg.inference_threads < 0:
        raise ValueError('inference_threads must not be negative')
    from .inference import signature
    signature(cfg)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    return cfg


def _clips(cfg: Config, args) -> list:
    paths = None
    if cfg.match and cfg.source_dir is None:
        paths = timeline.discover(cfg)
        base = Path(cfg.media_dir).resolve() / cfg.match
        paths = [p for p in paths if p.resolve().is_relative_to(base)]
        if not paths:
            raise ValueError(f"no videos in {base}")
    clips = timeline.build(cfg, paths)
    pat = getattr(args, "files", None)
    if pat:
        wanted = [w.strip() for w in pat.split(",") if w.strip()]
        clips = [c for c in clips if any(w in c.path.name for w in wanted)]
        if not clips:
            raise SystemExit(f"no media files matching {pat!r}")
    if getattr(args, "limit", None):
        clips = clips[: args.limit]
    if not clips:
        raise ValueError("no videos selected")
    return timeline.rebase(clips)


def _region_for(cfg: Config, clips, clip):
    """The field region belonging to this clip, if one was picked.

    A polygon saved next to the project as ``<file stem>.json`` wins: that is the
    per-file convention, and it is the most specific thing available.  Otherwise
    fall back to a region picked for this clip's recording, since the camera is
    repositioned between recordings and a polygon only applies to the files of
    the one it was drawn on.
    """
    import json as _json
    # fields/<stem>.json is the home for these; a file left in the project root
    # still works, so older setups keep running.
    for side in (pick.field_path(cfg, clip), cfg.source_path(cfg.fields_dir, clip.path) / f"{clip.path.stem}.json",
                 cfg.path(f"{clip.path.stem}.json")):
        if not side.exists():
            continue
        d = _json.loads(side.read_text())
        if d.get("field_polygon"):
            return {"ref_file": d.get("field_ref_file", clip.path.name),
                    "ref_time": d.get("field_ref_time", 60.0),
                    "polygon": d["field_polygon"], "source": str(side)}
    by_name = {c.path.name: c for c in clips}
    for r in cfg.regions():
        ref = by_name.get(r.get("ref_file"))
        if ref is not None and ref.chapter == clip.chapter:
            return r
    return None


def _mask_for(cfg: Config, clips, clip):
    """Field mask for a clip, using its own camera position."""
    region = _region_for(cfg, clips, clip)
    ref, poly = None, None
    if region:
        poly = region["polygon"]
        ref_clip = next((c for c in clips if str(c.path.resolve()) == region["ref_file"]
                         or c.path.name == region["ref_file"]), None)
        if ref_clip is not None:
            try:
                options={} if cfg.decode_gpu else dict(hwaccel=False)
                ref = grab_frame(ref_clip.path,region.get('ref_time',60.0),**options)
            except Exception:
                ref = None
    else:
        print(f"  WARNING: no field picked for {clip.path.name} "
              f"(recording {clip.chapter}); falling back to grass detection. "
              f"Run: scout pick")
    mask = detect.field_mask(cfg, clip, ref, poly)
    return mask


def _camera_features(cfg,clips,clip,det):
    """Project the saved reference in memory; never replace a saved field."""
    region=_region_for(cfg,clips,clip)
    reference_clip=(next((c for c in clips if str(c.path.resolve())==region['ref_file']
                         or c.path.name==region['ref_file']),None) if region else None)
    reference=None;keyframes=[]
    if reference_clip is not None:
        try:
            rw,rh=detect.decode_shape(reference_clip.info.width,reference_clip.info.height,640)
            reference=grab_frame(reference_clip.path,region.get('ref_time',60.),rw,rh,hwaccel=cfg.decode_gpu)
        except Exception:pass
    if reference is not None:
        times,frames=det.camera_times,det.camera_frames
        if times is None or frames is None:
            dw,dh=detect.decode_shape(clip.info.width,clip.info.height,cfg.decode_max_side)
            samples=[]
            indices={round(f*(max(1,round(clip.info.duration*cfg.sample_fps))-1)) for f in (0,.15,.325,.5,.675,.85,1)}
            import cv2
            for index in sorted(indices):
                when=min(index/cfg.sample_fps,max(0,clip.info.duration-.1))
                try:
                    image=grab_frame(clip.path,when,dw,dh,hwaccel=cfg.decode_gpu,
                        gpu_resize=bool(cfg.decode_max_side and cfg.decode_gpu))
                    cw,ch=detect.decode_shape(image.shape[1],image.shape[0],640)
                    samples.append((clip.start+when,cv2.resize(image,(cw,ch),interpolation=cv2.INTER_LINEAR)))
                except Exception:pass
            times=[t for t,_ in samples];frames=[rgb for _,rgb in samples]
        for when,rgb in zip(times,frames):
            polygon,quality=camera.project_field_region(reference,rgb,region['polygon'])
            keyframes.append(dict(at=float(when),polygon=polygon,**quality))
    # Compute the legacy mask only when it is needed; healthy projections use
    # already observed frames and avoid a second decode of the source.
    fallback=(np.ones(det.mask_shape,bool) if any(k['polygon'] is not None for k in keyframes)
              else _mask_for(cfg,clips,clip))
    filtered=camera.filter_detections(det,keyframes,fallback,cfg)
    return features.compute(cfg,filtered,np.ones(det.mask_shape,bool))


# --- commands -------------------------------------------------------------

def cmd_scan(args) -> int:
    cfg = _load(args)
    clips = _clips(cfg, args)
    print(timeline.describe(clips))
    for clip in clips:
        r = _region_for(cfg, clips, clip)
        print(f"  {clip.path}: {'field saved' if r else 'NO FIELD — run scout pick'}")
    return 0


def cmd_pick(args) -> int:
    cfg = _load(args)
    clips = _clips(cfg, args)
    if args.import_setup:
        pick.import_setup(cfg, clips, Path(args.import_setup))
    else:
        dst = pick.export_setup(cfg, clips, at=args.at,
                                open_browser=not args.no_browser)
        print(f"  Field setup: {dst}")
        print("  Draw each video's field, download setup.json, then run pick --import-setup PATH")
    return 0


def cmd_preview(args) -> int:
    """Render annotated frames at chosen times, to check the field and speeds."""
    cfg = _load(args)
    clips = _clips(cfg, args)
    times = [float(x) for x in args.at.split(",")] if args.at else [900.0]

    masks, dets, polys = {}, {}, {}
    for t in times:
        clip, _ = timeline.locate(clips, t)
        if clip.path.name in masks:
            continue
        d = detect.load_cached(cfg, clip)
        if d is None:
            print(f"  {clip.path.name}: not detected yet - run 'scout detect'")
            return 2
        dets[clip.path.name] = d
        masks[clip.path.name] = _mask_for(cfg, clips, clip)
        r = _region_for(cfg, clips, clip)
        if r:
            polys[clip.path.name] = r["polygon"]

    import cv2
    out_dir = cfg.path(cfg.out_dir) / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        img = report.annotate(cfg, clips, t, masks, dets, polygons=polys)
        dst = out_dir / f"frame_{int(round(t)):06d}s.jpg"
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {dst}")
    return 0


def cmd_renumber(args) -> int:
    """Put marker names into time order: Play 1, Play 2, ..."""
    n = groundtruth.renumber(args.mlt, prefix=args.prefix)
    print(f"  renamed {n} markers in {args.mlt} (backup: {args.mlt}.bak)")
    for m in groundtruth.load(args.mlt)[:3]:
        print(f"    {m.text:10s} {m.start:8.2f} -> {m.end:8.2f}")
    return 0


def _segment(cfg: Config, f, t_end, clf=None):
    """Plays from whichever segmenter is selected."""
    if cfg.segmenter == "model":
        if clf is None:
            path = model.resolve(cfg.path(cfg.model_path))
            if not path.exists():
                raise SystemExit(f"no model at {path} - run 'scout train' first")
            clf, names = model.load(path, cfg)
            from .model_features import names as feature_names
            expected = model.window_names([n for n in feature_names(f.names,cfg.model_feature_set)
                                           if n not in set(cfg.model_exclude or ())],cfg.model_offsets)
            if names != expected:
                raise ValueError("model feature schema changed; retrain from corrected reviews")
        return _plays_from_model(cfg, clf, f, t_end)
    return segment.find(cfg, f, t_end=t_end)


def _score_one(cfg: Config, mlt: str, clf=None):
    """Score one ground-truth project against the file it refers to."""
    name = Path(mlt).parent.name + "/" + Path(mlt).name
    clips = groundtruth.project_clips(mlt)
    gt = groundtruth.load(mlt)
    f = _features_for(cfg, clips)
    det = _segment(cfg, f, clips[-1].end, clf)
    return name, gt, det, evaluate.score(gt, det)


def _gt_files(cfg: Config, mlts: list[str]):
    """For each marked project: its file stem, features and marked plays."""
    out = []
    seen = set()
    for mlt in mlts:
        clips = groundtruth.project_clips(mlt)
        sources = {c.path.resolve() for c in clips}
        if seen & sources:
            raise ValueError("training projects must not reuse source videos; supply one corrected review per match")
        seen.update(sources)
        f = _features_for(cfg, clips)
        out.append((str(Path(mlt).resolve()), f, groundtruth.load(mlt), clips))
    return out


def _plays_from_model(cfg: Config, clf, f, t_end):
    prob = model.predict(clf, f, cfg)
    return model.to_plays(cfg,model.postprocess(cfg,f.t,prob),t_end)


def cmd_train(args) -> int:
    """Train the play model, and report how it does on files it never saw."""
    cfg = _load(args)
    print(f"  building dataset from {len(args.mlt)} marked file(s)...")
    per_file = _gt_files(cfg, args.mlt)
    ds = model.build(cfg, [(stem, f, marks) for stem, f, marks, _ in per_file])
    print(f"  {ds.summary()}\n")

    # Leave one file out: every score below comes from a model that never saw
    # the file it is scoring.  Anything else would flatter the result.
    print(f"  {'file':>14} {'recall':>8} {'precision':>10} {'F1':>7}  plays")
    hits = miss = fp = 0
    for gi, (stem, f, marks, clips) in enumerate(per_file) if len(per_file) > 1 else []:
        rows = ds.groups != gi
        clf = model.fit(cfg, ds, rows)
        plays = _plays_from_model(cfg, clf, f, clips[-1].end)
        sc = evaluate.score(marks, plays)
        hits += len(sc.hits); miss += len(sc.missed); fp += len(sc.false_pos)
        print(f"  {stem:>14} {sc.recall:8.1%} {sc.precision:10.1%} "
              f"{sc.f1:7.1%}  {len(plays):3d}")
    rec = hits / (hits + miss) if hits + miss else 0
    pre = hits / (hits + fp) if hits + fp else 0
    f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0
    if len(per_file) == 1:
        print("  One reviewed match: held-out evaluation unavailable.")
    else:
        print(f"\n  held-out overall   recall {rec:.1%} ({hits}/{hits+miss})   "
              f"precision {pre:.1%} ({hits}/{hits+fp})   F1 {f1:.1%}")

    clf = model.fit(cfg, ds)
    dst = cfg.path(args.out or cfg.model_path)
    model.save(dst, clf, ds.names, cfg)
    import hashlib
    import json
    from dataclasses import asdict
    provenance = {"config": asdict(cfg), "reviews": [
        {"path": str(Path(m).resolve()), "sha256": hashlib.sha256(Path(m).read_bytes()).hexdigest()}
        for m in args.mlt], "frames": len(ds), "groups": ds.files}
    dst.with_suffix(".training.json").write_text(json.dumps(provenance, indent=2))
    print(f"\n  final model trained on all files -> {dst}")

    print("\n  what the model leans on (drop in average precision when the")
    print("  signal is shuffled out entirely, all time offsets together):")
    for nm, w in model.importances(clf, ds.names, ds.X, ds.y, top=14):
        bar = "#" * int(round(w * 200))
        print(f"    {w:7.4f}  {nm:14s} {bar}")
    return 0


def cmd_eval(args) -> int:
    """Score the detector against one or more hand-marked projects."""
    cfg = _load(args)
    results = []
    for mlt in args.mlt:
        name, gt, det, sc = _score_one(cfg, mlt)
        results.append((name, gt, det, sc))
        print(f"\n  {name}   {sc.summary()}")
        if args.detail:
            print(f"    {'gt':>4} {'gt start':>9} {'det':>4} {'det snap':>9} "
                  f"{'overlap':>8}  result")
            for m in sc.matches:
                g = (f"{m.gt.index:4d} {m.gt.start:9.2f}" if m.gt
                     else f"{'-':>4} {'-':>9}")
                d = (f"{m.det.index:4d} {m.det.snap:9.2f}" if m.det
                     else f"{'-':>4} {'-':>9}")
                ov = f"{m.overlap:8.2f}" if m.kind == "hit" else f"{'':>8}"
                flag = {"hit": "ok", "missed": "MISSED",
                        "false": "FALSE POSITIVE"}[m.kind]
                print(f"    {g} {d} {ov}  {flag}")

    if len(results) > 1:
        hits = sum(len(r[3].hits) for r in results)
        miss = sum(len(r[3].missed) for r in results)
        fp = sum(len(r[3].false_pos) for r in results)
        rec = hits / (hits + miss) if hits + miss else 0
        pre = hits / (hits + fp) if hits + fp else 0
        f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0
        print(f"\n  combined            recall {rec:5.1%} ({hits}/{hits+miss})   "
              f"precision {pre:5.1%} ({hits}/{hits+fp})   F1 {f1:5.1%}")

    # Boundary accuracy is a separate question from finding the play at all.
    import numpy as _np
    ds = [m.det.start - m.gt.start for _, _, _, sc in results for m in sc.hits]
    de = [m.det.end - m.gt.end for _, _, _, sc in results for m in sc.hits]
    if ds:
        print(f"\n  clip boundaries on {len(ds)} found plays "
              f"(+ve = later than the marked one)")
        print(f"    start  median {_np.median(ds):+5.2f}s   mean |err| "
              f"{_np.abs(ds).mean():4.2f}s")
        print(f"    end    median {_np.median(de):+5.2f}s   mean |err| "
              f"{_np.abs(de).mean():4.2f}s")
    return 0


def cmd_tune(args) -> int:
    """Search for settings that match the marked footage."""
    cfg = _load(args)
    names = [groundtruth.media_for(m) or m for m in args.mlt]
    results = tune.search(cfg, args.mlt, quick=args.quick)
    best = tune.report(results, names, top=args.top,
                       key="worst_f1" if args.robust else "f1")

    print(f"\n  winner: F1 {best['f1']:.1%}  recall {best['recall']:.1%}  "
          f"precision {best['precision']:.1%}")
    for k, v in best["params"].items():
        print(f"    {k}: {v}")
    if args.write:
        import json as _json
        dst = Path(args.write)
        data = _json.loads(dst.read_text()) if dst.exists() else {}
        data.update(best["params"])
        dst.write_text(_json.dumps(data, indent=2))
        print(f"\n  written to {dst}")
    else:
        print("\n  (pass --write scout.json to apply these)")
    return 0


def cmd_collect(args) -> int:
    """Copy one game's plays into a single named folder."""
    cfg = _load(args)
    stems = [x.strip() for x in args.files.split(",") if x.strip()]
    rows = collect.run(cfg, args.game, stems, Path(args.out),
                       cfg.game_dir, dry_run=args.dry_run)
    total = sum(r.end - r.start for r in rows)
    where = Path(args.out) / args.game
    print(f"\n  {args.game}: {len(rows)} plays, {total/60:.1f} min of action")
    print(f"  from {len(set(r.source_file for r in rows))} camera file(s)")
    if args.dry_run:
        print("  (dry run - nothing copied)")
    else:
        print(f"  -> {where}")
        print(f"  -> {where/'plays.csv'}")
    return 0


def cmd_merge(args) -> int:
    """Join plays the camera split across a file boundary."""
    from .merge import merge as _merge
    nums = [int(x) for x in args.plays.split(",") if x.strip()]
    before, after = _merge(args.game, nums, Path(args.out))
    print(f"  joined plays {nums} of {args.game}")
    print(f"  {before} plays -> {after}, renumbered from {nums[0]}")
    return 0


def cmd_rename(args) -> int:
    """Rename a game's clips to a shorter prefix."""
    n = collect.rename(args.game, args.prefix, Path(args.out))
    print(f"  renamed {n} clips of {args.game} to {args.prefix}_001 ...")
    return 0


def cmd_rename_clips(args) -> int:
    from .rename_clips import rename
    cfg = _load(args)
    if not cfg.match and cfg.source_dir is None:
        raise ValueError('rename-clips requires --match or --source-dir')
    folder = cfg.path(cfg.out_dir) / 'clips'
    if not folder.is_dir():
        raise ValueError(f'clips folder not found: {folder}. Export the reviewed clips first '
                         f'with scout --match {cfg.match} cut')
    paths = [folder]
    if args.clips:
        if any(Path(name).name != name or '\\' in name for name in args.clips):
            raise ValueError('clip names must be filenames within the match clips folder')
        paths = [folder / name for name in args.clips]
    return rename(paths, args.pattern, args.start, args.pad, args.dry_run)


def cmd_gt(args) -> int:
    """Write the hand-marked plays out as a CSV, for reference and for tuning."""
    import csv as _csv
    cfg = _load(args)
    marks = groundtruth.load(args.mlt)
    out_dir = cfg.path(args.out or cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "stats.csv"

    def tc(t: float) -> str:
        m, sec = divmod(t, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    plays = groundtruth.to_plays(marks)
    with dst.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["index", "start_s", "end_s", "duration_s", "start_tc",
                    "end_tc", "gap_from_prev_s", "snap_to_snap_s", "label",
                    "clip_file"])
        prev_end = prev_start = None
        for m, pl in zip(marks, plays):
            gap = f"{m.start - prev_end:.2f}" if prev_end is not None else ""
            cad = f"{m.start - prev_start:.2f}" if prev_start is not None else ""
            w.writerow([m.index, f"{m.start:.2f}", f"{m.end:.2f}",
                        f"{m.duration:.2f}", tc(m.start), tc(m.end), gap, cad,
                        m.text, cut.name_for(pl)])
            prev_end, prev_start = m.end, m.start

    import numpy as _np
    import json
    durations = [m.duration for m in marks]
    summary = {"source": str(Path(args.mlt).resolve()), "plays": len(marks),
               "total_play_seconds": sum(durations),
               "mean_play_seconds": sum(durations) / len(durations) if durations else None}
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    if not marks:
        print(f"  0 plays -> {dst}")
        return 0
    d = _np.array([m.duration for m in marks])
    c = _np.diff([m.start for m in marks])
    print(f"  {len(marks)} plays -> {dst}")
    print(f"  duration  min {d.min():5.2f}  median {_np.median(d):5.2f}  max {d.max():5.2f}")
    if len(c):
        print(f"  snap gap  min {c.min():5.1f}  median {_np.median(c):5.1f}  max {c.max():5.1f}")
    print(f"  covers {marks[0].start:.1f}s .. {marks[-1].end:.1f}s")
    return 0


def cmd_strips(args) -> int:
    """Filmstrips for every play marked in a Shotcut project."""
    cfg = _load(args)
    marks = groundtruth.load(args.mlt)
    clips = groundtruth.project_clips(args.mlt)
    plays = groundtruth.to_plays(marks)
    out = Path(args.out or (cfg.path(cfg.out_dir) / "strips"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"  {len(plays)} marked plays from {args.mlt}")
    n = 0
    for p in plays:
        dst = out / f"{p.index:03d}_{int(p.snap):04d}s.jpg"
        if report.filmstrip(clips, p, dst, n=args.frames, width=args.width):
            n += 1
    sheet = report.snap_sheet(cfg, clips, plays, out / "_sheet.jpg", cols=3)
    print(f"  {n} filmstrips -> {out}")
    if sheet:
        print(f"  overview     -> {sheet}")
    return 0


def cmd_detect(args) -> int:
    cfg = _load(args)
    clips = _clips(cfg, args)
    _detect_for(cfg,clips)
    return 0


def _detect_for(cfg,clips):
    """Resolve the effective backend before reading or producing any source cache."""
    detector = None
    if any(detect.load_cached(cfg,c) is None for c in clips):
        detector=detect.create_detector(cfg)  # automatic initialization can select CPU
    print(f"  model {cfg.model} @ {cfg.imgsz}px, sampling {cfg.sample_fps} fps")
    for c in clips:
        detect.run(cfg, c, model=detector)


def _features_for(cfg: Config, clips) -> features.Features:
    cfg.validate_pipeline()
    if cfg.allow_gpu_fallback and cfg.model_pack and any(not detect.cache_path(cfg,c).is_file() for c in clips):
        # Standalone detect/segment jobs may be separate processes. Reuse a
        # complete, matching CPU cache set after automatic GPU initialization
        # failed; never combine different detector backends within a game.
        from dataclasses import replace
        from .model_packs import fallback_to_cpu
        cpu=replace(cfg);fallback_to_cpu(cpu)
        if all(detect.cache_path(cpu,c).is_file() for c in clips):
            cfg.__dict__.update(cpu.__dict__)
            print('  model pack: using complete CPU detection cache for segmentation',flush=True)
    parts = []
    for c in clips:
        det = detect.load_cached(cfg, c)
        if det is None:
            raise ValueError(f"{c.path}: missing detection cache; run scout detect with the same settings")
        parts.append(_camera_features(cfg,clips,c,det) if cfg.feature_pipeline=='dynamic-field-v1'
                     else features.compute(cfg,det,_mask_for(cfg,clips,c)))
    if not parts:
        raise SystemExit("no detections available")
    return features.Features(
        np.concatenate([p.t for p in parts]),
        np.concatenate([p.X for p in parts]),
        parts[0].names,
    )


def cmd_segment(args) -> int:
    cfg = _load(args)
    clips = _clips(cfg, args)
    missing = [c.path.name for c in clips if _region_for(cfg, clips, c) is None]
    if missing:
        print(f"  WARNING: no field picked for {', '.join(missing)}; "
              f"run 'scout pick'.\n")
    f = _features_for(cfg, clips)
    plays = _segment(cfg, f, clips[-1].end)
    out = cfg.path(cfg.out_dir)
    report.write_tables(plays, out)
    report.timeline_plot(cfg, f, plays, out / "review" / "timeline.png")
    total = sum(p.duration for p in plays)
    print(f"  {len(plays)} plays, {total/60:.1f} min of action "
          f"out of {(clips[-1].end - clips[0].start)/60:.1f} min of footage")
    print(f"  -> {out/'plays.csv'}   {out/'review'/'timeline.png'}")
    return 0


def cmd_cut(args) -> int:
    import hashlib
    cfg = _load(args)
    source = Path(args.mlt) if args.mlt else cfg.path(cfg.out_dir) / "review.mlt"
    def check_review():
        if args.review_hash is not None and hashlib.sha256(source.read_bytes()).hexdigest() != args.review_hash:
            raise ValueError('The review changed since the export preview. Reopen Export clips and try again.')
    check_review()
    clips = groundtruth.project_clips(source)
    plays = groundtruth.to_plays(groundtruth.load(source))
    check_review()
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        plays = [p for p in plays if p.index in wanted]
    paths = cut.cut_all(cfg, clips, plays, partial=bool(args.only),
                        name_pattern=args.name_pattern, start_number=args.start_number,
                        number_width=args.number_width)
    print(f"  {len(paths)} clips -> {cfg.path(cfg.out_dir) / 'clips'}")
    return 0


def _load_plays(cfg: Config) -> list:
    import json
    pj = cfg.path(cfg.out_dir) / "plays.json"
    if not pj.exists():
        raise SystemExit("no plays.json - run 'scout segment' first")
    # Results written before the current model are from a different detector.
    # Shipping those silently is worse than a noisy warning.
    mp = model.resolve(cfg.path(cfg.model_path))
    if cfg.segmenter == "model" and mp.exists():
        if pj.stat().st_mtime < mp.stat().st_mtime:
            print(f"  WARNING: {pj} is older than {mp} - these plays came from "
                  f"an earlier model.\n"
                  f"           Re-run 'scout segment' to bring them up to date.")
    return [segment.Play(**d) for d in json.loads(pj.read_text())]



def cmd_shotcut(args) -> int:
    cfg = _load(args)
    clips = _clips(cfg, args)
    plays = _load_plays(cfg)
    out = cfg.path(cfg.out_dir)
    dst = out / "review.mlt"
    if dst.exists() and not args.force:
        raise ValueError(f"{dst} exists; use --force only to replace the review")
    shotcut.write_review(cfg, clips, plays, dst)
    print(f"  {len(plays)} plays -> {dst}")
    return 0


def cmd_run(args) -> int:
    """Everything for a file: detect, find the plays, write the projects."""
    cfg = _load(args)
    clips = _clips(cfg, args)
    missing = [c for c in clips if _region_for(cfg, clips, c) is None]
    if missing:
        names = ", ".join(c.path.name for c in missing)
        print(f"  no field drawn for {names}.\n"
              f"  Run this first, once per camera position:\n"
              f"      scout --files {missing[0].path.stem} pick")
        return 2

    dst = cfg.path(cfg.out_dir) / "review.mlt"
    if dst.exists() and not args.force:
        raise ValueError(f"{dst} exists; use --force only to replace the review")
    _detect_for(cfg,clips)
    f = _features_for(cfg, clips)
    plays = _segment(cfg, f, clips[-1].end)
    shotcut.write_review(cfg, clips, plays, dst)
    out = cfg.path(cfg.out_dir)
    print(f"\n  done.  Open {out / 'review.mlt'} in Shotcut to check the plays.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scout",
                                 description="Cut flag football footage play by play")
    ap.add_argument('--model-pack', help='explicit installed pack: active or id@version')
    ap.add_argument('--model-packs-dir', help='model-pack store (default: application user data)')
    ap.add_argument("--match", help="imported game name or match subdirectory under media")
    ap.add_argument("--config", default=CONFIG_FILE, help="config JSON")
    ap.add_argument("--media-dir", dest="media_dir")
    ap.add_argument("--source-dir", help="use this recording folder in place")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--limit", type=int, help="only the first N files")
    ap.add_argument("--segmenter", choices=["rules", "model"],
                    help="how plays are found (default: model)")
    ap.add_argument("--files", help="only files whose name contains any of these "
                                    "(comma separated), e.g. --files DJI_0039")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser('model-pack', help='build, import, activate or roll back inference packs')
    s.add_argument('action', choices=['build', 'import', 'list', 'activate', 'rollback'])
    s.add_argument('--store')
    s.add_argument('--file')
    s.add_argument('--sha256')
    s.add_argument('--pack')
    s.add_argument('--approval')
    s.add_argument('--inputs')
    s.add_argument('--out')
    s.set_defaults(fn=cmd_model_pack)

    s = sub.add_parser("scan", help="list media and the timeline")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("pick", help="set up all video fields in one offline page")
    s.add_argument("--at", type=float, help="reference timestamp in each video")
    s.add_argument("--import-setup", help="import the downloaded setup JSON")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_pick)

    s = sub.add_parser("train", help="train the play model, with held-out scores")
    s.add_argument("--mlt", required=True, action="append",
                   help="marked project; repeat for every file you have")
    s.add_argument("--out", help="where to write the model")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="score detections against marked .mlt files")
    s.add_argument("--mlt", required=True, action="append",
                   help="ground-truth project; repeat for several files")
    s.add_argument("--detail", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("tune", help="search settings against marked .mlt files")
    s.add_argument("--mlt", required=True, action="append")
    s.add_argument("--quick", action="store_true", help="tiny grid, for a smoke test")
    s.add_argument("--robust", action="store_true",
                   help="rank by the weakest single file, not the average")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--write", help="write the winning settings to this config")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("collect", help="copy one game's plays into a named folder")
    s.add_argument("--game", required=True, help='e.g. "W0 01 NT17-SAI"')
    s.add_argument("--files", required=True,
                   help="camera files of this game, in order, comma separated")
    s.add_argument("--out", default="games", help="destination root")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("merge", help="join plays split across a camera file break")
    s.add_argument("--game", required=True)
    s.add_argument("--plays", required=True, help="consecutive numbers, e.g. 22,23")
    s.add_argument("--out", default="games")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("rename", help="rename a game's clips to a prefix")
    s.add_argument("--game", required=True)
    s.add_argument("--prefix", required=True, help="e.g. NT17_SAI")
    s.add_argument("--out", default="games")
    s.set_defaults(fn=cmd_rename)

    s = sub.add_parser('rename-clips', help='rename match clips with a custom <N> pattern')
    s.add_argument('--pattern', required=True, help='e.g. "xyz_<N>_abc"')
    s.add_argument('--start', type=int, default=1, help='first number (default: 1)')
    s.add_argument('--pad', type=int, help='number width (default: automatic from clip count)')
    s.add_argument('--clips', nargs='+', help='specific clip filenames, in desired order')
    s.add_argument('--dry-run', action='store_true', help='preview without renaming')
    s.set_defaults(fn=cmd_rename_clips)

    s = sub.add_parser("stats", aliases=["gt"], help="export marked plays from a .mlt as CSV")
    s.add_argument("--mlt", required=True)
    s.add_argument("--out", help="output directory")
    s.set_defaults(fn=cmd_gt)

    s = sub.add_parser("renumber", help="rename .mlt markers into time order")
    s.add_argument("--mlt", required=True)
    s.add_argument("--prefix", default="Play")
    s.set_defaults(fn=cmd_renumber)

    s = sub.add_parser("strips", help="filmstrips for plays marked in a .mlt")
    s.add_argument("--mlt", required=True, help="Shotcut project with markers")
    s.add_argument("--out", help="output directory")
    s.add_argument("--frames", type=int, default=8)
    s.add_argument("--width", type=int, default=420)
    s.set_defaults(fn=cmd_strips)

    s = sub.add_parser("preview", help="annotated frames at chosen timestamps")
    s.add_argument("--at", help="comma separated seconds, e.g. 640,760,903")
    s.set_defaults(fn=cmd_preview)

    s = sub.add_parser('export-classifier',help='export a trusted pickle classifier to numeric JSON')
    s.add_argument('--input',required=True)
    s.add_argument('--out',required=True)
    s.set_defaults(fn=cmd_export_classifier)

    s = sub.add_parser('export-detector',help='export detector weights into a new CPU inference directory')
    s.add_argument('--model',required=True)
    s.add_argument('--detector-backend',required=True,choices=['onnx','openvino'])
    s.add_argument('--out',required=True)
    s.add_argument('--device',default='cpu',choices=['cpu'])
    s.set_defaults(fn=cmd_export_detector)

    s = sub.add_parser("detect", help="run person detection (cached)")
    s.add_argument('--detector-backend', choices=['torch','onnx','openvino'])
    s.add_argument('--device', help='auto, cpu or a reference CUDA device')
    s.add_argument("--model"); s.add_argument("--sample-fps", dest="sample_fps", type=float)
    s.add_argument("--imgsz", type=int, help="inference resolution (default 1280)")
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("segment", help="find plays and write the timeline plot")
    s.set_defaults(fn=cmd_segment)

    s = sub.add_parser("cut", help="cut reviewed MLT markers into videos")
    s.add_argument("--cut-mode", dest="cut_mode", choices=["encode", "copy"])
    s.add_argument("--only", help="comma separated play numbers")
    s.add_argument("--mlt", help="cut the plays marked in this Shotcut project "
                                 "instead of the detected ones")
    s.add_argument("--name-pattern", help='filename pattern containing one <N>, e.g. "game_<N>"')
    s.add_argument("--start-number", type=int, default=1, help="first clip number (default: 1)")
    s.add_argument("--number-width", type=int, help="number width (default: automatic)")
    s.add_argument("--review-hash", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_cut)

    s = sub.add_parser("shotcut", help="write Shotcut .mlt projects")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_shotcut)

    s = sub.add_parser("run", help="detect plays and write review.mlt only")
    s.add_argument('--detector-backend', choices=['torch','onnx','openvino'])
    s.add_argument('--device', help='auto, cpu or a reference CUDA device')
    s.add_argument("--force", action="store_true")
    s.add_argument("--model")
    s.add_argument("--sample-fps", type=float)
    s.add_argument("--imgsz", type=int)
    s.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    for k in ("model", "sample_fps", "cut_mode", "file", "at", "only",
              "port", "no_browser", "chapter", "mlt", "out", "frames", "width",
              "prefix", "save", "strips", "detail", "quick", "robust",
              "top", "write", "game", "dry_run", "plays", "imgsz", "no_clips"):
        if not hasattr(args, k):
            setattr(args, k, None)
    try:
        return args.fn(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"scout: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nscout: cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
