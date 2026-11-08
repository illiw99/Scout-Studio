# Installer validation — 2026-11-08

Inspected GitHub Actions run
[42408776144](https://github.com/illiw99/Scout-Studio/actions/runs/42408776144)
at commit `084da27`. Windows, macOS arm64 and macOS x86-64 built successfully.
Linux stopped at source regression tests: the export integration test expected
pre/post padding `0.8/0.2` but briefly read `0.0/0.0`. All installed smoke-test jobs
were consequently skipped by the build-job dependency. These installers cannot
be called tested installations from this run.

An asynchronous review reload can run inside the export dialog's nested event
loop. It clears match data temporarily; the post-dialog refresh then read global
preferences instead of the game's preferences. The fix retains game controls
while data is pending and uses the dialog's captured game preferences after it
closes. A deterministic regression covers both pending-load cases.

Validation: all 11 native offscreen UI tests passed, including a real synthetic
queued MP4 export and the regression. Offscreen checks do not establish native
Windows/macOS installer execution. A fresh hosted run is required after pushing
the fix; no hosted success is claimed here.

The subsequent decision-only packaging integration passed the complete local
376-test check (two platform skips), source/wheel inspection and rebuilt native
Linux standalone smoke. Installer build inputs now reject generic detector
weights even through the legacy baseline override, and macOS distribution-root
links are handled explicitly. These local checks do not replace the pending
hosted installed-app smoke tests on all four platforms.

Local launcher installation now selects the checked frozen Linux application from
`dd482a2`. Fresh CPU export/detection and automatic CUDA detection both completed
on synthetic media. This app rebuild reused the previously validated dependency
runtime; it is not a fresh `.deb`, `.exe` or `.dmg` installer execution check.
