# Development

[README](../README.md) · [Installation](installation.md) · [Code map](code-map.md)

PlayChop uses the `scout` import package and `scout-flag` distribution. The CLI
and native application share the same detection, review, and export pipeline.
Keep the application in Python and native Qt.

## Set up an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[desktop,dev]'
```

The `desktop` extra supplies PySide6. The `dev` extra supplies build tooling and
Python 3.10's optional TOML parser. Tests use the standard-library `unittest`.

On minimal Ubuntu systems, install Qt's graphics runtime before running desktop
tests, including offscreen tests:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y libegl1 libgl1 libopengl0
```

GitHub Actions installs these libraries in the Python 3.12 desktop job. The
Python 3.10 job checks the CLI without the optional desktop extra.

## Run checks

```bash
PYTHON=.venv/bin/python bash scripts/check.sh
python scripts/check-package.py
```

The first command checks agent configuration, CLI help, and the test suite. Qt
checks use the offscreen platform by default. The packaging check builds a
source distribution, builds its wheel, and checks the wheel outside the source
tree. It uses temporary directories and does not download dependencies or touch
library data.

For focused tests:

```bash
python -m unittest discover -s tests -p 'test_desktop_review.py' -v
```

Offscreen tests do not verify native display behavior, audible playback, codecs,
or GPU acceleration. Smoke-test the affected native behavior when changing it.
Use generated temporary video fixtures rather than personal match recordings.

## Build packages

```bash
python -m build
```

This writes the wheel and source archive to `dist/`. Version metadata comes from
`scout.__version__`; update it in one place when preparing an actual release.
`MANIFEST.in` specifies source archive contents. Wheels contain Python code,
Qt SVG assets, the typing marker, and package metadata.

Models, YOLO weights, footage, fields, saved libraries, local configuration,
credentials, caches, and generated exports are excluded. A wheel is a Python
package, not a standalone operating-system installer. Do not publish artifacts
without explicit authorization.

## Build standalone installers and releases

Linux CI installs CPU-only PyTorch before application dependencies to avoid
downloading CUDA libraries on hosted test runners. Desktop regression tests need
the PulseAudio client library (`libpulse0`) even with Qt's offscreen backend;
installer build runners also install ALSA (`libasound2`).

The installer spec explicitly collects TorchVision's native operator libraries,
including modern `_C_stable` and `image_stable` extension names. Python import
analysis alone misses these dynamically loaded files. The installed-app smoke
check executes native NMS; failures print their JSON report and retain a
`smoke-process.log` diagnostic artifact.

The **Standalone installers** GitHub workflow runs on pull requests, pushes to
`main`, version tags and manual dispatch. It builds Windows x64, Ubuntu x64,
macOS Apple Silicon and macOS Intel independently using Python 3.12 and
PyInstaller. Python, Qt, detection dependencies and FFmpeg are included; user
data, detector weights and trained classifiers are excluded.

Each installer is downloaded into a fresh runner with no checkout or application
dependencies. CI installs it, launches the installed executable from a temporary
working directory, verifies dependency imports, synthetic video generation and
decoding, clip export, CLI child dispatch, the native window and video playback,
then uninstalls it. Linux uses Xvfb. JSON reports and installer logs are retained
as diagnostic artifacts. Builds also run the source regression suite on Linux.

For a local build on the target operating system, use a separate build environment:

```bash
python -m pip install -r packaging/build-requirements.txt
python -m pip install '.[desktop]'
python scripts/build-installers.py
```

Use the workflow's CPU Torch and Intel macOS compatibility constraints when
reproducing a CI build. Windows needs Inno Setup, Linux `dpkg-deb`, and macOS
`hdiutil`. Outputs are in `dist/installers/`. `scripts/test-installer.py` changes
system installation state: run it only on disposable CI machines. It takes the
downloaded installer artifact directory and does not need the source package.

To release, update `scout.__version__` and the changelog, commit, and push a
matching tag such as `v1.0.0`. The workflow rejects mismatched tags. After all four
builds and installation checks pass, it creates a GitHub Release with the four
installers and `SHA256SUMS.txt`. Branch and manual runs provide Actions artifacts
without publishing a release. Only the release job has repository write
permission. Tag pushes publish automatically, so push release tags deliberately.

Windows code signing and Apple Developer signing/notarization are not configured.
Add those credentials and signing steps before distributing to environments that
require trusted publishers. CI does not bypass or validate operating-system
download security prompts.

## Make changes

EditorConfig supplies UTF-8, LF line endings, final newlines, and four-space
Python indentation. Markdown keeps intentional hard-break whitespace.

Start with the [code map](code-map.md). Keep changes focused; avoid moving public
imports or changing review/cache contracts as incidental cleanup. Follow
[AGENTS.md](../AGENTS.md) when using development agents.

- Make regular commits after each coherent change and its relevant checks.
- Describe what changed and why; include useful validation in the commit body.
- Use the configured Git author, without `Co-authored-by` trailers or assistant attribution.
- Preserve unrelated edits and never stage footage, credentials, or generated state.
- Do not amend, squash, rewrite history, push, or publish unless requested.

The [agent workflow](multi-agent-workflow.md) describes optional delegation.

## Runtime data

A library may live outside the source checkout. Its `.scout-library.json` index
remembers imported game folders. Each game owns a `Scout/` folder for reviews,
fields, detection caches, exports and processing logs, shared by CLI and GUI.
Copied footage can still live under `media/<match>`. Workspace `out/`, `fields/`
and `.scout-cache/` are legacy migration inputs; new default jobs do not create
them. Optional `scout.json` settings and trained `models/` remain separate from
redistributable application packages. Explicit custom destinations are honored.

### Rename old game data folders

Only the visible `Scout/` folder is used for game data. Close Studio and stop
processing before migrating an older `.scout/` or lowercase `scout/` folder:

```bash
python scripts/migrate-state.py '/path/to/Game' --dry-run
python scripts/migrate-state.py '/path/to/Game' --apply --backup-dir '/path/to/new-backup'
```

Pass additional game folders to migrate them together. The tool validates the
whole batch, refuses duplicate data folders or linked paths, saves original
metadata in the new backup directory, updates saved references, and renames the
directories. Video and cache files are moved without encoding or copying. A
failed migration rolls the batch back; `result.json` records its outcome.
The tool never merges existing `Scout`, `.scout` or `scout` folders. Resolve any
collision before retrying. Choose game folders, not the repository root containing
the Python `scout` package.
