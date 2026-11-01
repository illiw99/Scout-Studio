# Scout

Scout finds plays in flag-football recordings. Set up the field for each video,
generate a Shotcut review, correct its range markers, then export clips or
statistics. The saved `review.mlt` is the source of truth.

## Scout Studio — native desktop app

Scout Studio provides a native desktop window for the complete match workflow.
Open **Scout Studio** from the Linux Applications menu after the one-time setup
below. Day-to-day use needs no terminal or Python knowledge.

- Browse and search matches in a local library, or create a new match.
- Import camera videos with the file picker or drag them into the window.
  Imports copy the originals, report progress, and reject filename collisions.
- Select each video, scrub to a reference frame, and click around its field.
  Drag corners to adjust, right-click to remove, then **Save field**.
- Click **Find plays**. The processing queue runs one match at a time while you
  keep browsing. It also observes an existing W3 batch and waits for it to finish.
  **Queue** (Ctrl+0) opens the single app-wide queue in a separate, nonblocking
  panel; starting a job keeps you on the match you are viewing.
- Watch the **entire match** in **Review & export**, including all footage between
  detected plays. Playback automatically continues into the next camera file.
  Scrub anywhere on the full-match slider or marker timeline. The timeline shows
  a sliding 60-second window by default; selecting a play focuses on its range
  with context so the handles stay easy to grab. **Scroll** slides through time;
  **Ctrl+scroll** zooms around the pointer. Choose 15s/30s/60s/2min/5min windows,
  drag the scrollbar, or use **Fit match** for the overview.
- Review puts the video first, with a compact play inspector at the side.
  **Alt+6** hides the inspector for more viewing space; **Ctrl+B** toggles the match
  library. The video retains its aspect ratio as the workspace resizes.
- **Shift + drag** on the timeline creates and saves a play at the length you
  select, in either direction and across camera-file boundaries. A live preview
  shows its bounds and duration; **Esc** cancels. Default labels use the start
  time, such as `Play 00:12:34.560`, and follow boundary changes. Custom names
  are kept. Click a play and drag either handle to resize it; undo is available.
- Click a marker to jump to it. Adjust start/end times and labels, drag the
  selected marker's timeline edges (snapped to source frames), add missing plays,
  or delete proposals. Playback previews a dragged boundary, pauses during a
  scrub, and resumes afterward only if it was playing beforehand.
  **Save play changes** writes the authoritative
  `review.mlt`; the original is backed up as `review.before-studio.mlt`.
- To mark a missed play, press **M** at its start, use **I** / **O** to set in/out
  points at the cursor, and press **Ctrl+S** to save. **M** edits an existing play
  when the cursor is inside its range. Right-click a timeline play for **Edit**,
  **Replay**, or **Delete**, or select it and press **Delete**. Undo restores it.
- Playback and timeline shortcuts follow [Shotcut's reference](https://www.shotcut.org/howtos/keyboard-shortcuts/).
  **Space** plays/pauses even when a header button has focus. **J/K/L** reverse,
  pause, and play forwards; repeated J or L increases speed through **10×**.
  Space and pause controls reset speed to **1×**.
  Reverse uses paused frame seeks across source files. **Left/Right** step frames;
  holding **K** and tapping **J/L** also steps frames. **Page Up/Down** seek one
  second, with **Shift** for two, **Ctrl** for five, or **Ctrl+Shift** for ten.
  **< / >** navigate plays; **\** repeats and **Ctrl+M** mutes. The speed menu
  supports **0.1×–10×**; Scout's **Alt+− / Alt+=** slow down / speed up.
- **Ctrl+`** opens a normal, resizable video window (or brings it back).
  **F11**, **Ctrl+Shift+F**, or **F** toggle video fullscreen; **Esc** returns to
  the normal window. Double-click the video for fullscreen. Speed and the current
  match stay synchronized between windows. **Ctrl+0** opens Jobs; **Ctrl+Q** quits.
  Shortcuts keep normal text-field behavior; **Ctrl+S** works while editing.
  Unsaved edits prompt before leaving a marker. Film Hub's navy, gray and lavender
  palette is available in **Light mode** and **Dark mode**, remembered between
  launches. No Film Hub server is required.
- Playback, marker and timeline controls use compact SVG symbols with accessible
  names and action-only tooltips. Shortcut hints appear only in **?**; press
  **?**, **/**, or **F1** for the scrollable guide. **=/−** zoom, **0** fits the
  match, and **Ctrl+Shift+P** centers on the cursor. Exact position, window length
  and marker bounds stay visible.
- Use **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z) for marker additions, deletions,
  boundary changes and label edits. Saves and undo/redo preserve your viewing
  position and playback. Undo history lasts until you reload or change matches.
- Camera files follow natural filename order, including edited suffixes such as
  `0077a`. For an older saved review, **… → Order camera files by name** moves
  markers with their original source footage and saves an exact original backup.
  A play split across files that become separated is kept as separate ranges.
- **Export clips** creates MP4s from the saved review. The **…** menu holds
  refresh, output-folder and optional Shotcut actions. The review workflow is built in.

One-time setup for a source checkout:

```bash
.venv/bin/python -m pip install -e '.[desktop]'
.venv/bin/python scripts/install_studio.py
```

You can also launch with `./start-studio.sh` or `scout-studio`. Select another
library with **Change library**; the app remembers your choice. A library is a
folder containing `media/`, `fields/`, `out/`, and optional `scout.json` settings.
New libraries use this checkout's included play model. The first detection run
may download YOLO weights. Large imports need space for a copy of the footage.

The desktop dependency is optional; existing command-line usage is unchanged.
The Linux menu launcher uses this checkout's installed environment. This is not
a standalone installer for other computers; those need the one-time setup too.

## Install

Python 3.10+ is required. Run from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scout --help
```

Install Shotcut separately if you prefer it for reviewing. Scout uses system FFmpeg when available,
otherwise the binary supplied by `imageio-ffmpeg`. Detection runs on CPU or CUDA;
the first detection run may download YOLO weights. The included play model is at
`models/play.pkl`. Keep that file with this checkout; a wheel does not bundle it.
Use `--segmenter rules` to generate initial proposals without a trained model.

## One match, start to finish

Put each match's camera files in its own directory:

```text
media/
  Match-01/
    DJI_0001.MP4
    DJI_0002.MP4
```

Global options such as `--match`, `--media-dir`, and `--out-dir` go **before** the
command. Match names are directory names, not filename searches.

### 1. Set up the fields

```bash
scout --match Match-01 scan
scout --match Match-01 pick
```

This opens **one offline HTML page for all videos**. No webserver runs. Select a
video from the dropdown, click around its field, and repeat for every video.
Drag points to adjust them; right-click removes a point. Existing polygons are
loaded for adjustment. Click **Download match setup**, then import the downloaded
file:

```bash
scout --match Match-01 pick --import-setup ~/Downloads/setup.json
```

Fields are saved separately for each source video, including videos with the same
filename in different matches. `pick --at 30` chooses another reference timestamp.
For a remote/headless machine, use `pick --no-browser`, copy the generated
`out/Match-01/field-setup.html` to your computer, and transfer the downloaded JSON
back for import. The page contains its reference images and works offline.

### 2. Generate and review

```bash
scout --match Match-01 run
```

Detection is cached. The only review deliverable is
`out/Match-01/review.mlt`: no automatic MP4 export or second MLT. Review it inside
Scout Studio's **Review & export** tab, or open it in Shotcut and add, delete,
rename, or adjust **range markers**, then save. Review the entire match: unmarked footage becomes negative
training data. Point markers are treated as notes and ignored.

Keep the original footage timeline intact. Trimming/splitting clips, gaps,
multiple video tracks, and speed changes are unsupported. Scout checks source
order and clip bounds before cutting or training. It exports original footage,
not Shotcut effects or overlays. See [Shotcut's marker instructions](https://www.shotcut.org/blog/new-release-211031/).

An existing review is protected. `run --force` explicitly replaces it with new
predictions, discarding marker corrections; save a separate copy first if needed.
Source videos must remain at the paths stored in the project, or be relinked in
Shotcut.

### 3. Export the reviewed result

```bash
scout --match Match-01 cut
scout --match Match-01 stats --mlt out/Match-01/review.mlt
```

`cut` reads the match's `review.mlt` by default. Use `cut --mlt PATH` for a renamed
review. Each range becomes an MP4 in `out/Match-01/clips/`, including ranges across
camera-file boundaries. Re-encoding is the default; `--cut-mode copy` is faster
but cuts at keyframes. `--only 1,3` exports a subset.

Full exports remove obsolete clips tracked by `clips/exports.json` after all new
clips render successfully. Failed rendering returns a nonzero exit status and
leaves the previous export intact. Untracked files are preserved. Partial exports
retain other clips; use a full export after deleting or renumbering markers.

`stats` writes `stats.csv` with boundaries, duration, labels, spacing, and clip
names, plus `stats.json` with play count and total/mean duration. These are timing
statistics; team/player attribution and scoring are not inferred. Empty reviews
produce zero clips and valid empty statistics.

## Improve the model from corrected reviews

```bash
scout train \
  --mlt out/Match-01/review.mlt \
  --mlt out/Match-02/review.mlt \
  --out models/improved.pkl
```

Training reuses cached person detections for **every video in each review**, then
recomputes field-based features and labels them from the corrected markers. It
retrains the play classifier; it does not fine-tune YOLO. Include earlier reviewed
matches each time to retain their examples. Source footage and saved fields are
still needed. Duplicate source videos across training projects are rejected to
avoid misleading validation scores.

With two or more independent reviews, Scout evaluates by holding out whole
reviews. With one review it trains without claiming a held-out score. Training
requires both play and non-play examples. A `.training.json` file records the
configuration and review hashes beside the model.

To use the new model, create `scout.json`:

```json
{"model_path": "models/improved.pkl"}
```

Use the same detection settings for generation and training. Missing caches fail
with an instruction to run `detect`; changing source files, sample rate, detection
model, resolution, or confidence invalidates their cache. The stronger cache
identity in this version intentionally invalidates old Scout caches once.
Load only trusted `.pkl` models. Models are Python/scikit-learn artifacts; retain
the training environment or retrain when upgrading dependencies.

## Files and maintenance

```text
fields/                    per-video field polygons
.scout-cache/              reusable person detections
models/                    play classifiers and training provenance
out/Match-01/
  field-setup.html          offline setup page
  review.mlt                editable review, authoritative for export/training
  clips/                    reviewed MP4s and export manifest
  stats.csv, stats.json     optional timing statistics
```

`--media-dir /path/to/project/media` resolves cache, fields, and default output
relative to `/path/to/project`. `--files TEXT` and `--limit N` can narrow a match;
use the same selection when importing its setup. Legacy `fields/<stem>.json`
polygons still load, but importing setup writes path-specific field files.
Additional diagnostic commands are listed by `scout --help`; `gt` aliases `stats`.

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

Tests cover the MLT workflow, field setup, cache identity, retraining inputs,
export failures, and real FFmpeg cuts across source boundaries. Detection quality
still needs validation on your own camera position and reviewed matches.
