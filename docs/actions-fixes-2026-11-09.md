# GitHub Actions corrections — 2026-11-09

Inspected the failed workflows at `a5441a9`:

- [Installer run 42460587710](https://github.com/illiw99/Scout-Studio/actions/runs/42460587710):
  Windows failed before PyInstaller analysis because a catalogue asset no longer
  matched its recorded byte size. Linux and both macOS builds succeeded. The
  shared build dependency skipped every installed smoke-test job.
- [Test run 42460587695](https://github.com/illiw99/Scout-Studio/actions/runs/42460587695):
  Python 3.10 failed `test_run_only_review`; Python 3.12 passed.

Windows Git's automatic LF-to-CRLF conversion changed checked-out JSON manifests.
Model catalogue sizes and hashes identify exact bytes, so `/models/** -text`
now preserves those bytes on every platform. The new regression checks out all
catalogued assets using Git with `core.autocrlf=true` into a temporary directory
and verifies the original catalogue. Integrity checks remain strict.

The synthetic workflow tests replace detector, feature and segmentation stages,
but newly discoverable production defaults were preparing real detector weights
before those stubs ran. The minimal Python 3.10 environment has no ONNX exporter.
Those fixtures now isolate default discovery and explicitly reject any weight
fetch or graph export. Production model selection remains covered separately by
catalogue, default-profile and runtime tests; inference dependencies are not
added to the minimal test environment to hide the issue.

Local focused validation: 18 workflow tests and six catalogue tests passed.
Source/wheel checks passed, including allowlisted assets and installed CLI smoke.
The Git attributes file is included in source archives. Native Windows execution
requires the next hosted build; local checkout simulation is not native evidence.
