# Code map

Navigation hints, not a substitute for current source. Update affected entries
when moving responsibilities; read only paths relevant to the task.

| Concern | Start here |
| --- | --- |
| CLI commands and pipeline orchestration | `scout/cli.py`, `scout/__main__.py` |
| Configuration, per-game/source state paths and tunable defaults | `scout/config.py` |
| Shared CLI/GUI game-folder index, validation and atomic persistence | `scout/storage.py` |
| Source discovery (including linked folders and .scout exclusion), clip order and match timeline | `scout/timeline.py` |
| FFmpeg discovery, probing and frame decoding | `scout/media.py` |
| YOLO detection, device choice and persistent caches | `scout/detect.py` |
| Field masks, camera movement and polygon setup | `scout/field.py`, `scout/pick.py` |
| Player tracking and motion features | `scout/track.py`, `scout/features.py` |
| Heuristic play segmentation | `scout/segment.py` |
| Trained classifier, evaluation and parameter tuning | `scout/model.py`, `scout/evaluate.py`, `scout/tune.py` |
| MP4 cuts, cross-source clips, audio and encoding | `scout/cut.py` |
| Export filename patterns and native naming preview | `scout/clip_names.py`, `scout/desktop/export_dialog.py` |
| Shotcut review/playlist writing and marker parsing | `scout/shotcut.py`, `scout/groundtruth.py` |
| Explicit source reordering with marker remapping and backups | `scout/review_order.py` |
| Tables, timelines, filmstrips and annotations | `scout/report.py` |
| Collection, merging and clip renaming | `scout/collect.py`, `scout/merge.py`, `scout/rename_clips.py` |
| Native app entry, window, field/review editing and shortcuts | `scout/desktop/__init__.py`, `scout/desktop/app.py` |
| Qt task runners, click-to-seek slider and field canvas | `scout/desktop/widgets.py` |
| Native match playback and marker timeline | `scout/desktop/review.py` |
| Resizable video window, fullscreen transitions and subtle in-play video frame | `scout/desktop/player_window.py` |
| Timestamp play labels and custom-name preservation | `scout/desktop/labels.py` |
| Persistent game-folder index, in-place/copy imports, per-game .scout paths, fields and conflict-safe review saves | `scout/desktop/library.py` |
| Background command queue and cancellation | `scout/desktop/jobs.py` |
| Native theme and app icon | `scout/desktop/theme.py`, `scout/desktop/icons.py`, `scout/desktop/scout.svg` |
| Packaging, entry points, dependencies and release contents | `pyproject.toml`, `MANIFEST.in`, `LICENSE` |
| Temporary source/wheel builds and artifact validation | `scripts/check-package.py`, `tests/test_packaging.py` |
| Standalone installer builds, install checks and tag releases | `packaging/`, `scripts/build-installers.py`, `scripts/test-installer.py`, `.github/workflows/installers.yml` |
| Frozen GUI/CLI dispatch and installed app/media smoke test | `scout/desktop/bundle.py`, `tests/test_bundle.py` |
| User installation, Studio and CLI guides | `docs/installation.md`, `docs/studio.md`, `docs/cli.md` |
| Development, validation and contribution conventions | `docs/development.md`, `CONTRIBUTING.md` |
| Studio launcher and desktop installation | `start-studio.sh`, `scripts/install_studio.py` |
| Pipeline/export/decode regression tests | `tests/test_workflow.py`, `tests/test_export.py`, `tests/test_decode.py` |
| Detector precision compatibility and cache invalidation | `tests/test_precision.py` |
| Custom clip names, collision handling and export dialog checks | `tests/test_clip_names.py`, `tests/test_desktop_export_dialog.py` |
| Camera filename order and review marker migration checks | `tests/test_review_order.py` |
| Native library, playback and UI checks | `tests/test_desktop_library.py`, `tests/test_desktop_review.py`, `tests/test_desktop_ui.py`, `tests/test_desktop_jobs.py`, `tests/test_desktop_labels.py` |
| Rename regression checks | `tests/test_rename_clips.py` |
| Agent workflow and local checks | `AGENTS.md`, `.codex/config.toml`, `.codex/agents/`, `scripts/check-agents.py`, `scripts/check.sh` |

CLI / native Qt -> ordered media timeline -> cached detection -> features -> play
segmentation -> Shotcut review and MP4 export. The native UI edits fields and
review markers, and queues existing CLI commands as subprocesses.

Runtime state includes imported game folders and their `.scout/` state,
`media/`, `games/`, configuration and trained `models/`. Legacy `fields/`,
`.scout-cache/` and `out/` are migration inputs. Treat those as user data even when ignored
by Git. Tests use temporary fixtures; do not validate by modifying a real library.
Read README only for setup and operational details. The existing CLI setup helper
in `scout/pick.py` is separate from the native desktop workflow.
