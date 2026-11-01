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

## Make changes

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

A library may live outside the source checkout. Typical library folders include `media/`, `fields/`,
`.scout-cache/`, `models/`, `out/`, and optional `scout.json` settings. Existing
paths and data remain authoritative. Models intentionally remain separate from
redistributable application packages.
