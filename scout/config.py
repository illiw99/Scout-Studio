"""Tunable parameters for the whole pipeline.

Everything that might need adjusting for a different camera position, field or
league sits here.  ``Config.load`` merges a JSON file over the defaults so a
recording can be tuned without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from .storage import STATE_DIR_NAMES, state_directory


@dataclass
class Config:
    # --- input -------------------------------------------------------------
    match: str | None = None
    media_dir: str = "media"
    #: An exact recording folder; match remains its display/registry name.
    source_dir: str | None = None
    out_dir: str = "Scout"
    #: Persistent detections stored with the game.
    cache_dir: str = "Scout/cache"
    #: Hand-drawn field polygons, one `<file stem>.json` per camera file.
    fields_dir: str = "Scout/fields"
    #: Measure how players are arranged along the field rather than along the
    #: image.  Measured worth: +0.5 points of F1, and both recall and precision
    #: improved.  The three venues so far sit at +13, +17 and -17 degrees from
    #: horizontal, so "spread across the field" was already being measured along
    #: the wrong direction; this stops a tilted camera breaking it outright.
    field_axis: bool = True
    extensions: tuple[str, ...] = (".mp4", ".mov", ".mts", ".avi", ".mkv")

    #: Files whose size is still changing are being copied in; skip them until
    #: the size has been stable for this many seconds.
    stability_seconds: float = 3.0

    #: Two consecutive files belong to the same continuous recording when the
    #: second starts within this many seconds of the first one ending.  DJI
    #: splits at 4 GB and leaves a gap well under a second.
    chapter_gap_tolerance: float = 5.0

    # --- sampling ----------------------------------------------------------
    #: Frames per second fed to the detector.  5 fps resolves a snap to 200 ms,
    #: which is finer than the padding applied to clips anyway.
    sample_fps: float = 5.0

    # --- detection ---------------------------------------------------------
    model: str = "yolo11m.pt"
    #: Inference resolution.  Players are 60-100 px tall in a 1080p frame, so
    #: downscaling below 1280 starts to lose the far side of the field.
    imgsz: int = 1280
    conf: float = 0.25
    device: str = "auto"          # "auto" | "cuda" | "cuda:0" | "cpu"
    batch: int = 8
    detector_backend: str = 'torch'  # reference | explicit onnx/openvino CPU engine
    inference_threads: int = 0     # 0 lets the exported CPU runtime choose
    inference_cache_dir: str | None = None  # defaults to application user cache
    half: bool = True             # legacy switch; False disables fp16
    quantize: str | int | None = "fp16"  # fp16/fp32, 16/32, or None

    model_pack: str | None = None  # explicit id@version or active
    model_packs_dir: str | None = None

    # --- field region ------------------------------------------------------
    #: Polygon of the playing surface in normalised (0-1) frame coordinates.
    #: ``None`` means "estimate it from the green playing surface", which finds
    #: every pitch at the venue - pick it by hand with ``scout pick``.
    field_polygon: list[list[float]] | None = None
    #: File and timestamp the polygon was picked on, so drift in later files can
    #: be measured against it.
    field_ref_file: str | None = None
    field_ref_time: float = 60.0
    #: The camera may be repositioned between recordings, and then one polygon
    #: will not do.  Each entry is {"ref_file", "ref_time", "polygon"} and
    #: applies to every file of the recording its reference frame came from.
    field_regions: list[dict] | None = None

    def regions(self) -> list[dict]:
        """All field regions, including a legacy single polygon."""
        out = list(self.field_regions or [])
        if self.field_polygon and not any(
                r.get("ref_file") == self.field_ref_file for r in out):
            out.insert(0, {"ref_file": self.field_ref_file,
                           "ref_time": self.field_ref_time,
                           "polygon": self.field_polygon})
        return out
    #: Ignore anything above this line (normalised y): spectators, benches and
    #: the neighbouring pitches sit along the top of the frame.
    horizon_y: float = 0.30
    #: Push the top edge of the field down by this fraction of frame height.
    #: Only meant for an *estimated* field, where the boundary lands exactly
    #: where substitutes and spectators stand.  A hand-drawn polygon is used as
    #: drawn, so this stays at zero: shrinking it throws away plays run at the
    #: far end of the field, which is where the boundary matters most.
    field_shrink_top: float = 0.0
    #: Detections smaller/larger than this (normalised height) are not players
    #: on this field - they are background crowd or a foreground passer-by.
    min_person_h: float = 0.030
    max_person_h: float = 0.160

    # --- play segmentation -------------------------------------------------
    #: A play needs roughly two units of five plus officials on the field.
    min_players: int = 6
    max_players: int = 20
    #: Speed in body-heights per second.  The camera looks along the field at a
    #: shallow angle, which compresses apparent motion, so these are far smaller
    #: than a top-down view would give: standing is ~0.05, a play peaks near 1.
    sprint_speed: float = 0.90
    #: A track counts as a player once it has moved this fast at some point in
    #: the last few seconds.  Below it, it is someone watching from the line.
    active_speed: float = 0.45
    #: A snap is recognised by its shape: enough players, spread out in a
    #: formation, standing still, then breaking together.
    #: How wide the players must be spread (fraction of frame width).  Low,
    #: because a play run at the far end of the field looks narrow from here:
    #: perspective compresses it.  The stillness and the simultaneous break do
    #: most of the discriminating.
    formation_span: float = 0.15
    #: At most this many may be moving during the set.
    still_max: int = 3
    #: How long the set must hold, and how far back to look for it.
    still_hold: float = 0.8
    #: At least this many must break within `burst_window` of each other.
    burst_min: int = 4
    burst_window: float = 1.0
    #: How far back to look for the real start of a play, from the point the
    #: burst was recognised.
    max_backoff: float = 3.0
    #: No two snaps closer together than this.  Measured snap-to-snap gaps in
    #: the marked footage are 25-49 s, so 10 s is safely below a real cadence
    #: while still suppressing a re-fire during the same play.
    refractory: float = 10.0
    #: Fraction of players sprinting that sustains a play once started.
    action_hi: float = 0.30
    action_lo: float = 0.13
    #: A play ends once action stays below ``action_lo`` this long.
    end_hold: float = 1.2
    #: Bursts closer together than this are one play (scramble, lateral).
    merge_gap: float = 1.5
    min_duration: float = 1.5
    max_duration: float = 22.0

    #: Seconds kept before the snap and after the whistle.  Fitted to
    #: hand-marked footage: after back-dating the snap, the residual lead-in
    #: people actually want is about a second, and almost no tail.
    pre_roll: float = 1.2
    post_roll: float = 0.3

    # --- learned segmenter ---
    #: Where the trained model lives.
    model_path: str = "models/play.pkl"
    #: "rules" is the hand-tuned state machine, "model" the learned one.
    #: Held out over 190 marked plays: rules 74.8% F1, model 95.8%.
    segmenter: str = "model"

    #: Features the model may not use.  Anything that encodes the camera setup
    #: rather than the game - how big players look, how far away they are - lets
    #: a model score well on the venues it was trained on and fail on a new one.
    #: Measured: excluding these costs 0.6 points of F1 on the tournament the
    #: model was trained on, and buys a model that cannot be keying on camera
    #: distance or on how fast a particular game is played.  Worth it, since
    #: every new location is a new camera setup.
    model_exclude: tuple[str, ...] = (
        "height_med", "depth_med", "depth_far", "quiet_age")

    #: Kept deliberately small: ten files from two venues is not much variety,
    #: and a larger model would memorise this tournament rather than the game.
    model_max_depth: int = 5
    model_max_iter: int = 250
    model_learning_rate: float = 0.06
    model_min_leaf: int = 60
    model_l2: float = 1.0

    #: A play opens where the probability clears `model_threshold` and stays
    #: open while it holds above `threshold * low_ratio`, so a dip mid-play does
    #: not split it in two.
    model_threshold: float = 0.50
    model_low_ratio: float = 0.55
    model_smooth: float = 0.6
    model_min_duration: float = 2.0
    #: Padding around what the model calls the play.  Both are zero because the
    #: learned interval already lands on the marked boundaries: measured over
    #: 184 held-out plays, padding of 0.6/0.4 s pushed the start 0.66 s early and
    #: the end 0.50 s late.  Unlike the rule-based segmenter, which found only
    #: the burst and needed padding to reach the real edges of the play, the
    #: model is trained on the whole interval and predicts it directly.
    model_pre_roll: float = 0.0
    model_post_roll: float = 0.0

    # --- output ------------------------------------------------------------
    #: "encode" is frame accurate; "copy" is near instant but snaps starts to
    #: the nearest keyframe (up to ~1 s early on DJI footage).
    cut_mode: str = "encode"
    crf: int = 20
    preset: str = "veryfast"
    clip_workers: int = 4

    @property
    def root(self) -> Path:
        """The project directory, whatever directory the command was run from.

        Taken as the parent of the media folder, so pointing --media-dir at a
        project elsewhere picks up that project's fields, cache and output
        rather than creating empty ones beside the shell's current directory.
        An exact source_dir uses the game folder itself as its root.
        """
        if self.source_dir is not None:
            return Path(self.source_dir).resolve()
        media = Path(self.media_dir)
        return media.parent.resolve() if media.is_absolute() else Path.cwd()

    @property
    def game_dir(self) -> Path:
        """The exact source folder that owns this game's application state."""
        if self.source_dir is not None:
            return Path(self.source_dir).resolve()
        media = Path(self.media_dir).resolve()
        return media / self.match if self.match else media

    def path(self, value: str | Path) -> Path:
        """Resolve a configured path against the project directory."""
        p = Path(value)
        if p.is_absolute():
            # Older desktop job configs may name the game-local state directly.
            for name in STATE_DIR_NAMES:
                base = self.game_dir / name
                if p.is_relative_to(base):
                    return state_directory(self.game_dir) / p.relative_to(base)
            return p
        if p.parts and p.parts[0] in STATE_DIR_NAMES:
            return state_directory(self.game_dir).joinpath(*p.parts[1:])
        return self.root / p

    def source_path(self, value: str | Path, source: Path) -> Path:
        """Locate default state for a source, including multi-game training."""
        p = Path(value)
        if p.is_absolute() or not p.parts or p.parts[0] not in STATE_DIR_NAMES:
            return self.path(p)
        source = Path(source).absolute()
        if (self.source_dir is not None or self.match) and source.is_relative_to(self.game_dir):
            return state_directory(self.game_dir).joinpath(*p.parts[1:])
        # A reviewed game can live anywhere and contain nested camera folders.
        for folder in source.parents:
            # A collection folder itself may be named Scout. It is not state
            # when it contains the source video being located.
            if any((folder / name).is_dir() and not source.is_relative_to(folder / name)
                   for name in STATE_DIR_NAMES):
                return state_directory(folder).joinpath(*p.parts[1:])
        media = Path(self.media_dir).resolve()
        if source.is_relative_to(media):
            relative = source.relative_to(media)
            folder = media / relative.parts[0] if len(relative.parts) > 1 else media
        else:
            folder = source.parent
        return state_directory(folder).joinpath(*p.parts[1:])

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            return cfg
        data = json.loads(p.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise SystemExit(f"{p}: unknown config keys: {', '.join(sorted(unknown))}")
        for k, v in data.items():
            setattr(cfg, k, v)
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
