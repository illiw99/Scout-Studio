# Scout Studio

**Scout Studio** is a native desktop application for reviewing flag-football
film, finding plays, and exporting clips. Watch the full match, refine play
boundaries on the timeline, and keep your work in a portable `review.mlt` file.

The application is written in Python and PySide6. Daily review happens in a
normal desktop window; a command-line interface is available for automation.

## AI-generated, human-guided

**Scout Studio's own code, tests, automation, and documentation are fully
AI-generated, guided by its human maintainer.** The maintainer provides the idea,
requirements, priorities, and feedback from using the app; AI assistants generate
and revise the implementation. This statement describes the project's own work.
Third-party libraries and tools have their own authors and licenses.

The idea started with a practical need: spend less time scrubbing through long
flag-football recordings to find and cut plays. Scout combines person detection
and a trained play/non-play classifier to propose useful segments. The native
review interface lets a person check the match, catch missed plays, refine
boundaries, and export the clips they want.

This is also an experiment in building a useful desktop tool through human-led
AI development. Predictions are suggestions to review, and the project continues
to evolve through testing and feedback.

## What you can do

- Import camera files and browse matches in a local library.
- Draw field regions, detect plays, and review all footage between them.
- Move and resize plays with frame snapping, automatic saves, and undo.
- Use Shotcut-style controls, a zoomable timeline, and resizable video windows.
- Export MP4 clips with a custom `<N>` filename pattern and a live preview.
- Run processing through one background queue while continuing to review.

## Start Scout Studio

For installation without Python, use the Windows, Linux or macOS installers
produced by the [release workflow](docs/development.md#build-standalone-installers-and-releases).
See [standalone installation](docs/installation.md#standalone-installers) for downloads
and platform requirements.

From a source checkout, using Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[desktop]'
python -m scout.desktop
```

For a Linux Applications-menu shortcut, run this once in that environment:

```bash
python scripts/install_studio.py
```

Then open **Scout Studio** from Applications. You can also use `scout-studio` or
`./start-studio.sh`. See the [installation guide](docs/installation.md) for
platform notes, command-line-only installs, and optional NVIDIA decoding.

The source checkout includes a trained play classifier in `models/play.pkl`.
The first person-detection run may download YOLO weights. Releases separate
application code from trained models, footage, and your library data.

## Documentation

| Guide | Contents |
| --- | --- |
| [Scout Studio](docs/studio.md) | Import, field setup, playback, timeline editing, saving, and export |
| [Installation](docs/installation.md) | Source setup, optional desktop dependency, FFmpeg, and CUDA |
| [Command-line workflow](docs/cli.md) | Detection, MLT reviews, clips, statistics, renaming, and training |
| [Development](docs/development.md) | Project layout, tests, package builds, and contribution workflow |
| [Future path](docs/future-path.md) | Proposed architecture, smaller inference runtime, model packs, and tested release promotion |
| [Code map](docs/code-map.md) | Find the source responsible for each feature |
| [Change history](CHANGELOG.md) | Features and maintenance changes |

## Command-line use

The Python distribution is `scout-flag`; the import package and CLI remain `scout`.
Install with `python -m pip install -e .` if you only need the CLI.

```bash
scout --help
scout --match Match-01 run
scout --match Match-01 cut --name-pattern 'Saints_<N>_offense'
```

Set up the match's field regions before detection. The saved `review.mlt` is the
source of truth for export and training; Shotcut remains an optional editor.

## Repository layout

```text
scout/             Detection, media pipeline, and command-line tools
  desktop/         Native Qt application and packaged visual assets
tests/             Unit tests and temporary-video integration checks
docs/              User and developer guides
scripts/           Validation and local desktop installation tools
pyproject.toml     Package metadata, dependencies, and entry points
```

Footage, field polygons, detection caches, trained models, and exports stay in
your library. Repository cleanup and package builds do not move or rewrite them.

Python 3.10+ · Native PySide6 desktop · FFmpeg · [MIT license](LICENSE)
