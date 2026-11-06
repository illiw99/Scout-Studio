# Code map

Navigation hints, not a substitute for current source. Update affected entries
when moving responsibilities; read only paths relevant to the task.

| Concern | Start here |
| --- | --- |
| CLI commands and pipeline orchestration | `scout/cli.py`, `scout/__main__.py` |
| Configuration, per-game/source state paths and tunable defaults | `scout/config.py` |
| Shared CLI/GUI game-folder index, validation and atomic persistence | `scout/storage.py` |
| Explicit `.scout`/`scout` to `Scout` migration, backups and rollback | `scout/state_migration.py`, `scripts/migrate-state.py`, `tests/test_state_migration.py` |
| Source discovery (including linked folders and Scout exclusion), clip order and match timeline | `scout/timeline.py` |
| FFmpeg discovery, probing and frame decoding | `scout/media.py` |
| Background helper launch options and Windows console suppression | `scout/processes.py`, `tests/test_processes.py` |
| YOLO detection, device choice and persistent caches | `scout/detect.py` |
| Explicit CPU graph runtimes, preprocessing/NMS and managed detector exports | `scout/inference.py`, `tests/test_inference.py` |
| Windows GPU availability and faster detector candidates | `docs/detection-performance.md` |
| Isolated model/runtime speed comparisons and local measurements | `scripts/benchmark-detectors.py`, `scripts/prepare-scaled-detector-features.py`, `docs/detector-benchmark-2026-11-05.md` |
| Offline Studio defaults and approved standalone model/media inputs | `scout/model_defaults.py`, `docs/default-model.md`, `tests/test_model_defaults.py`, `tests/test_default_profile.py` |
| Immutable inference packs, compatibility checks and activation rollback | `scout/model_packs.py`, `tests/test_model_packs.py`, `docs/model-packs.md` |
| Manual release checks and verified installer staging | `scout/updates.py`, `tests/test_updates.py`, `docs/updates.md` |
| Field masks, camera movement and polygon setup | `scout/field.py`, `scout/camera.py`, `scout/pick.py`; isolated experiments: `scripts/prepare-camera-region-features.py` |
| Player tracking and motion features | `scout/track.py`, `scout/features.py` |
| Heuristic play segmentation | `scout/segment.py` |
| Reviewed-game improvement protocol and strict interval metrics | `docs/model-improvement.md`, `scout/model_evaluation.py`, `scripts/evaluate-reviewed-games.py`, `scripts/calibrate-play-thresholds.py`, `scripts/extract-video-features.py`, `scripts/combine-video-features.py`, `tests/test_model_evaluation.py` |
| Frozen research heads, complete-pipeline timing and reserved full-game confirmation | `scripts/freeze-reviewed-candidate.py`, `scripts/benchmark-reviewed-pipeline.py`, `scripts/evaluate-frozen-candidate.py`, `tests/test_reviewed_game_evaluation.py` |
| Native adaptive-field profiles, resized decode/cache coordinates and verified GPU references | `scout/cli.py`, `scout/detect.py`, `scout/media.py`, `scout/inference.py`, `tests/test_decode_profile.py`, `tests/test_reference_assets.py` |
| Trained classifier, derived play features, evaluation and parameter tuning | `scout/model.py`, `scout/model_features.py`, `scout/evaluate.py`, `scout/tune.py` |
| NumPy-only exported classifier inference and numerical/boundary checks | `scout/classifier_runtime.py`, `tests/test_classifier_runtime.py` |
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
| Persistent game-folder index, in-place/copy imports, per-game Scout paths, fields and conflict-safe review saves | `scout/desktop/library.py` |
| Registry-only index backup/import, writer locking and standard user-data defaults | `scout/storage.py`, `scout/desktop/app.py`, `tests/test_library_index_import.py` |
| Background command queue and cancellation | `scout/desktop/jobs.py` |
| Versioned local worker jobs, progress events and verified results | `scout/worker.py`, `tests/test_worker.py`, `tests/test_runtime_boundary.py` |
| Native theme, application identity and app icons | `scout/desktop/theme.py`, `scout/desktop/icons.py`, `scout/desktop/identity.py`, `scout/desktop/scout.svg`, `scout/desktop/scout.ico`, `scripts/build-icons.py` |
| Frozen runtime profile and reduced-bundle validation | `scout/runtime.py`, `tests/test_runtime_profile.py`, `docs/runtime-validation-2026-11-05.md` |
| Packaging, entry points, dependencies and release contents | `pyproject.toml`, `MANIFEST.in`, `LICENSE` |
| Temporary source/wheel builds and artifact validation | `scripts/check-package.py`, `tests/test_packaging.py` |
| Standalone installer builds, disk-image retries, install checks and tag releases | `packaging/`, `scripts/build-installers.py`, `scripts/test-installer.py`, `tests/test_installer_build.py`, `.github/workflows/installers.yml` |
| Candidate manifests, installed-test evidence and promotion without rebuilding | `scripts/release-candidate.py`, `tests/test_release_candidate.py`, `.github/workflows/promote-installers.yml` |
| Frozen GUI/CLI dispatch and installed app/media smoke test | `scout/desktop/bundle.py`, `tests/test_bundle.py` |
| User installation, Studio and CLI guides | `docs/installation.md`, `docs/studio.md`, `docs/cli.md` |
| Development, validation and contribution conventions | `docs/development.md`, `CONTRIBUTING.md` |
| Proposed architecture and delivery roadmap | `docs/future-path.md` |
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

Runtime state includes imported game folders and their `Scout/` state,
`media/`, `games/`, configuration and trained `models/`. Legacy `fields/`,
`.scout-cache/` and `out/` are migration inputs. Treat those as user data even when ignored
by Git. Tests use temporary fixtures; do not validate by modifying a real library.
Read README only for setup and operational details. The existing CLI setup helper
in `scout/pick.py` is separate from the native desktop workflow.
