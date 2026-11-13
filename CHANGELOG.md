# Changelog

This file summarizes user-visible changes. Focused Git commits retain the
implementation and validation history.

## Unreleased

## 1.0.3 — 2026-11-13

- Publish only native installers; first-launch setup downloads pinned upstream
  Python/processing packages into private CPU/CUDA environments, with repair
  and cancellation. No separate runtime or support releases are required.

- Ship the saved visual-only boundary refiner as the default decision pack 0.5.0;
  retain the prior classifier for rollback and Transformer compatibility.
- Refined cuts reproduce 73.48% joint ±1-second accuracy on four confirmation
  games, with unchanged 92.05% precision / 98.78% recall; targets remain unmet.
- Use small validated numeric weights without training dependencies; prevent
  refinement from reversing plays or creating new overlaps.
- Add separate refiner-overhead and bounded full-pipeline CPU/GPU benchmarks.

## 1.0.2 — 2026-11-09

- Smaller native installers on Windows, macOS and Linux, with the same
  first-launch hardware-aware processing-engine setup wizard.
- Separate verified CPU/CUDA engines and native macOS CPU/Metal support, with
  cancellable downloads, CPU fallback and cache repair.
- Keep review/export available offline and run heavy model operations in the
  downloaded worker. Preserve existing job, model and review integrity checks.
- Publish runtime parts below GitHub's per-file limit; bind every part to native
  install-test evidence and the installer's embedded catalogue.

## 1.0.1 — 2026-11-09

Tag created; installer publication was blocked by GitHub's 2 GiB Windows asset limit.

### Models and runtime

- Ship the approved play decision model and selected Person Transformer with
  matching configuration. Download verified YOLO weights on first detection;
  prepare and cache optimized CPU exports when needed.
- Add persistent pre/post-play export padding and automatic clip numbering.
- Unify native form controls, colors and spacing; compact playback controls and
  improve timeline drag previews.
- Preserve game export preferences during asynchronous review reloads.
- Preserve exact model bytes in Windows checkouts and isolate synthetic tests
  from production model downloads and exports.

### Native review

- Linux checkout launchers open their own workspace, and startup recognises a
  local game index before a remembered default library.
- Empty game folders can be imported and display a clear no-videos status;
  refresh the match after adding footage.
- Windows background FFmpeg tasks run without opening extra console windows,
  while retaining progress and error logs in Scout.
- Scout Studio provides match browsing, MP4 import, field setup, and full-match playback.
- The video can fill the workspace or enter fullscreen.
- Shotcut-style controls support frame stepping, reverse/forward shuttle, and 0.1×–10× speeds.
- The timeline defaults to 30 seconds and retains the chosen zoom when selecting plays.
- Shift-drag creates variable-length plays; dragging the body moves a play without resizing it.
- Clicking the first or second half pauses at the corresponding play boundary.
- A softly animated lavender video border indicates that the cursor is inside a play.
- Timeline edits save atomically with backups, conflict detection, and undo/redo.

### Processing and exports

- A single background queue handles processing across matches.
- Camera files follow natural filename order; existing reviews can be reordered with marker remapping.
- Export applies `<N>` filename patterns with automatic numbering and fixed
  pre/post-play padding, with saved preferences.
- Exported clips can also be renamed with pattern, selection, and dry-run options.
- CUDA decoding falls back to CPU without duplicating frames already delivered.
- Detector precision supports current and legacy Ultralytics arguments while preserving default caches.

### Repository and packaging

- Windows executable, setup and shortcuts use the Scout logo, with a shared
  taskbar identity and installed-icon verification.
- Application, installer, shortcut and release branding consistently use
  Scout Studio, with Scout-Studio filenames.
- macOS installer creation retries transient disk-image lock failures, verifies
  the image before publishing it, and preserves build diagnostics.
- Game data now lives in the visible `Scout/` folder. An explicit migration tool
  renames older `.scout/` and lowercase `scout/` folders with metadata backups.
- Native Windows, Ubuntu and macOS installers bundle Python, Qt and FFmpeg.
- GitHub Actions tests fresh installations, playback/export and removal before
  publishing installers and checksums for a matching version tag.
- User guides are separated from installation, CLI, and development instructions.
- Package metadata, MIT license, source manifest, and artifact validation are explicit.
- The optional native app supports `scout-studio` and `python -m scout.desktop`.
- Build checks allow only selected project-trained decision models and exclude
  generic detector weights, credentials, footage, caches and library state.
- Agent instructions require regular focused commits without coauthor attribution.
