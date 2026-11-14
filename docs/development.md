# Development

[README](../README.md) · [Installation](installation.md) · [Code map](code-map.md)

Scout Studio uses the `scout` import package and `scout-flag` distribution. The CLI
and native application share the same detection, review, and export pipeline.
Keep the application in Python and native Qt.

The [future path](future-path.md) records the proposed architecture and delivery
roadmap. The commands below describe today's implementation.

## Set up an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[desktop,dev,training,inference]'
```

The `desktop` extra supplies PySide6. `reference` supplies the original
Ultralytics/PyTorch detector and legacy pickle classifier dependencies;
`training` adds training/evaluation and diagnostic plots. `inference` supplies
optional exported-model runtimes. The base package supports media, saved reviews
and exports without training frameworks. The `dev` extra supplies build tooling and
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
Qt assets, the typing marker, package metadata and explicitly allowlisted
project-trained decision models.

Generic detector weights, arbitrary model files, footage, fields, saved libraries, local configuration,
credentials, caches, and generated exports are excluded. A wheel is a Python
package, not a standalone operating-system installer. Do not publish artifacts
without explicit authorization.

## Build standalone installers and releases

The **Standalone installers** workflow builds Windows x64, Linux x64 and both
Mac architectures on PRs, `main` pushes and manual dispatch. Release CI uses
`bootstrap`: native Qt/media support, our decision models and small hash-pinned
engine recipes. Processing packages are fetched upstream into private Python;
they are not frozen into large release runtime archives. See the
[delivery contract](runtime-delivery-contract.md) and [lock regeneration](../packaging/locks/README.md).

On the target OS, use a disposable isolated build environment and the platform's
hash-pinned dependency snapshot. Follow `.github/workflows/installers.yml` for
exact prerequisites (Inno Setup, `dpkg-deb` or `hdiutil`; Linux Qt/PulseAudio/ALSA
libraries). Build all native recipes before freezing:

```bash
python scripts/build-runtime.py --engine-id linux-x86_64-cpu --output-dir dist/installers
python scripts/build-runtime.py --engine-id linux-x86_64-cuda --output-dir dist/installers
python scripts/build-installers.py --runtime-profile bootstrap --runtime-files dist/installers
```

Use matching Windows engine ids, or the single native macOS id. `reference`
and `inference` remain explicit developer build profiles. `test-installer.py`
changes system installation state: run only on disposable machines. CI downloads
candidates into fresh runners without the checkout, installs upstream recipes,
then tests GUI/playback/export/worker/probes, upgrade and uninstall preservation.
Hosted tests cannot establish physical Windows GPU speed or every codec/display.
macOS retries only `hdiutil` resource-busy errors, verifies the final image and
retains disk-image logs. Native NMS checks catch missing dynamic TorchVision libs.

Candidate manifests bind version, commit, repository, workflow run/attempt,
platform, dependency/build metadata, installer/helper hashes and installed-test
evidence. Raw build artifacts last 30 days, tested candidates 90 days subject to
repository retention. The upgrade baseline defaults to pinned 1.0.2; repository
variables can override it. CI authenticates the fixed baseline release API lookup
without forwarding tokens on redirects. See [upgrade policy](updates.md).

For release, update `scout.__version__`/changelog, push the commit to `main`, wait
for its complete four-build/four-installed-test run, then push the authorized
version tag at that same commit. Promotion selects only a successful push-to-main
run for that exact repository/commit. PR/manual builds cannot be promoted.
Current-attempt job/step evidence must be complete; a partial rerun is insufficient.
Expired artifacts require rebuilding/retesting; tag promotion never rebuilds.

Promotion uses the tagged verifier, treats downloaded helpers as data, validates
all hashes/test bindings and publishes exactly four unchanged installers. Compact
candidate evidence lives in release notes, matched by the updater to GitHub asset
digests. No checksum/manifest/runtime attachments or support releases are needed.
Old releases retain their original metadata. Existing Studio ≤1.0.2 needs a manual
installer download once for the metadata transition. Only promotion has repository
write permission; creating a tag still requires user authorization.

`tests/test_release_candidate.py` and related fixture tests exercise rejection
paths without installing applications. Real native build/install/upgrade results
are linked from [model status](model-status.md); do not describe local offscreen
coverage as platform/hardware evidence. Minimum supported Windows/macOS versions
remain incompletely established.

Windows code signing and Apple Developer signing/notarization are not configured.
Add those credentials and signing steps before distributing to environments that
require trusted publishers. CI does not bypass or validate operating-system
download security prompts.

Windows app/setup executables embed `scout/desktop/scout.ico`, compiled from the
existing SVG logo at shell icon sizes. After changing the SVG, regenerate it with
`python scripts/build-icons.py` (desktop extra required). Package checks require
both assets, and the Windows installation check compares the executable's
embedded icon resources against the bundled ICO. The app and shortcuts share
the `org.scoutstudio.studio` taskbar identity.

## Make changes

The native queue now writes version1 job descriptors and unique per-job config
snapshots, then launches `scout.worker` (or the frozen `--worker-job` dispatcher).
The worker adapts the existing CLI commands, emitting versioned log/progress/
completion events and a job-bound result. The queue requires matching success
evidence as well as a normal zero process exit. Cancellation still kills the
process tree; missing/truncated results never count as success. No training
framework is imported to parse a job or start the worker protocol. The installed
smoke test exercises this exact path through a synthetic export.

Binary numeric histogram-boosted classifiers can be saved by `scout train --out
model.json`. This validated tree format runs through NumPy without scikit-learn
or pickle. Export checks reference probabilities on deterministic probes;
regressions also compare NaN routing and resulting play intervals at `1e-12`
probability tolerance. Feature ordering, excluded features, sample rate and
temporal offsets remain checked on load. Categorical/multiclass exports are
rejected. Existing `.pkl` models remain supported through the `reference` extra;
load only trusted pickle files. Production-model equivalence must still be
checked before selecting an exported model as an approved baseline pack.

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
