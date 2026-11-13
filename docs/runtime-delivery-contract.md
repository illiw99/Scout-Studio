# Processing engine delivery contract

Scout Studio 1.0.3 bootstrap installers contain native Qt, media/export support,
project decision models and a small platform-specific pip recipe catalogue. They
exclude the processing frameworks. Source installations retain their explicit
Python environment. Runtime Python is pinned to 3.12.12; the application supports
Python 3.10+ for source installations.

Catalogue schema 2 has `application_version`, `python`, `target`, `bootstrap`,
`files` and `runtimes`. `bootstrap` pins the upstream uv 0.9.7 platform wheel URL,
size, SHA256 and exact ZIP executable member. `packaging/bootstrap-uv.json` records
those pins from PyPI. `files` maps simple filenames to size/SHA256 identities for
one Scout wheel, `worker_launcher.py` and native `requirements-<engine>.txt` files.
Every runtime maps `requirements`, `wheel` and `entrypoint` to these files. All
recipe files are embedded adjacent to `runtime-catalog.json` in the installer.
No generic dependency binary, archive part or support release is published.

Windows/Linux catalogues contain native CPU and CUDA recipes; macOS contains one
native CPU/Metal recipe. Requirements preserve exact native wheel pins, hashes
and upstream indexes from `packaging/locks/`, removing Qt, installer/build tooling
and duplicate `opencv-python` while retaining `opencv-python-headless`.

Setup verifies uv, provisions a private managed Python, and uses pip-compatible
uv commands to install only hash-pinned upstream wheels into a private virtual
environment. It never changes a system Python or graphics driver. Network access
is required for initial setup and repair. Cancellation, progress, integrity
verification and successful worker probes precede publication of readiness.

The worker runs as `<private-python> -I worker_launcher.py` followed by
`--worker-job DESCRIPTOR`, `--scout-cli ARGS...` or `--engine-probe REPORT.json`.
Adjacent `engine-id.json` binds the engine id, requirements hash and Scout wheel
hash before Scout imports. Probes verify CPU NMS, classifier, ONNX inference,
available OpenVINO and bundled pipeline compatibility; CUDA builds must probe on
GPU-less CI. Native installed smoke prepares every platform recipe from upstream,
then exercises the GUI, worker, export and uninstall paths.

Candidate manifests bind the installer, catalogue and each recipe file to the
same build and installed-test evidence. Promotion verifies all four candidates
before copying only the four unchanged deb/dmg/exe installers into the main
release. Candidate metadata, checksums and probe reports remain in Actions
artifacts. The legacy schema-1 archive helper remains for compatibility fixtures.
