# Changelog

This file summarizes user-visible changes. Focused Git commits retain the
implementation and validation history. The package version is currently 1.0.0;
these changes have not been published as a release.

## Unreleased

### Native review

- Scout Studio provides match browsing, MP4 import, field setup, and full-match playback.
- The video can fill the workspace, move to a resizable window, or enter fullscreen.
- Shotcut-style controls support frame stepping, reverse/forward shuttle, and 0.1×–10× speeds.
- The timeline defaults to 30 seconds and retains the chosen zoom when selecting plays.
- Shift-drag creates variable-length plays; dragging the body moves a play without resizing it.
- Clicking the first or second half pauses at the corresponding play boundary.
- A softly animated lavender video border indicates that the cursor is inside a play.
- Timeline edits save atomically with backups, conflict detection, and undo/redo.

### Processing and exports

- A single background queue handles processing across matches.
- Camera files follow natural filename order; existing reviews can be reordered with marker remapping.
- Export offers a live `<N>` filename preview, starting numbers, padding, and saved preferences.
- Exported clips can also be renamed with pattern, selection, and dry-run options.
- CUDA decoding falls back to CPU without duplicating frames already delivered.
- Detector precision supports current and legacy Ultralytics arguments while preserving default caches.

### Repository and packaging

- Native Windows, Ubuntu and macOS installers bundle Python, Qt and FFmpeg.
- GitHub Actions tests fresh installations, playback/export and removal before
  publishing installers and checksums for a matching version tag.
- User guides are separated from installation, CLI, and development instructions.
- Package metadata, MIT license, source manifest, and artifact validation are explicit.
- The optional native app supports `scout-studio` and `python -m scout.desktop`.
- Build checks exclude models, credentials, footage, caches, and library state.
- Agent instructions require regular focused commits without coauthor attribution.
