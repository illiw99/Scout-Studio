# Scout

Scout finds plays in flag-football recordings. Set up the field for each video,
generate a Shotcut review, correct its range markers, then export clips or
statistics. The saved `review.mlt` is the source of truth.

## Install

Python 3.10+ is required. Run from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
scout --help
```

Install Shotcut separately for reviewing. Scout uses system FFmpeg when available,
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
`out/Match-01/review.mlt`: no automatic MP4 export, second MLT, or internal review
application. Open it in Shotcut and add, delete, rename, or adjust **range
markers**, then save. Review the entire match: unmarked footage becomes negative
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
