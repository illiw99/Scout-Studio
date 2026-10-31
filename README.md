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
- Watch the **entire match** in **Review & export**, including all footage between
  detected plays. Playback automatically continues into the next camera file.
  Scrub anywhere on the full-match slider or marker timeline. The timeline shows
  a sliding 60-second window by default; selecting a play focuses on its range
  with context so the handles stay easy to grab. **Scroll** slides through time;
  **Ctrl+scroll** zooms around the pointer. Choose 15s/30s/60s/2min/5min windows,
  drag the scrollbar, or use **Fit match** for the overview.
- Click a marker to jump to it. Adjust start/end times and labels, drag the
  selected marker's timeline edges (snapped to source frames), add missing plays,
  or delete proposals. Playback previews a dragged boundary, pauses during a
  scrub, and resumes afterward only if it was playing beforehand.
  **Save play changes** writes the authoritative
  `review.mlt`; the original is backed up as `review.before-studio.mlt`.
- To mark a missed play, press **N** at its start, use **I** / **O** to set in/out
  points at the cursor, and press **Ctrl+S** to save. **Space** plays/pauses,
  **Left/Right** skip five seconds, and **comma/period** step one frame while
  paused. Film Hub's local-player behavior is adapted in Python/Qt: **R** replays,
  **L** repeats the selected play (or whole match), **M** mutes, **F** opens native
  fullscreen (Esc exits), and speeds range from 0.25× to 3×. Shortcuts except
  Ctrl+S are inactive while typing. Unsaved edits prompt before leaving a marker.
  Film Hub's shared navy, gray and lavender palette is available in **Light mode**
  and **Dark mode**, remembered between launches. No Film Hub server is required.
- Playback, marker and timeline controls use compact SVG symbols with accessible
  names and shortcut tooltips. Press **?** or **F1** for the shortcut guide.
  **Page Up/Down** jump between plays, **Shift+Left/Right** skip one second,
  **+/−** zoom, **Ctrl+0** fits the match, **C** centers on the cursor, and **[ / ]**
  slide half a window. Exact position, window length and marker bounds stay visible.
- Use **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z) for marker additions, deletions,
  boundary changes and label edits. Saves and undo/redo preserve your viewing
  position and playback. Undo history lasts until you reload or change matches.
- **Export play clips** creates MP4s from the saved review. **Open output folder**
  takes you to the result. Shotcut is optional; the review workflow is built in.

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
