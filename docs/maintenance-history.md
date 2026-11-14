# Engineering history and documentation retention

Current behavior belongs in the user/developer guides. This page preserves the
important lessons from superseded progress logs without presenting them as open
work. Full prior wording remains in Git at commit `ccb4808`, for example
`git show ccb4808:docs/actions-fixes-2026-11-09.md`. Raw private benchmark reports,
source/GT snapshots and model checkpoints were not deleted by this cleanup.

| Milestone / correction | Retained lesson and evidence |
| --- | --- |
| Initial runtime reduction,5November | Isolated reference/inference-only builds validated NumPy classifier parity, worker/CPU graphs and reduced dependency payloads. Historic bundle sizes are not current upstream-download installer sizes |
| Native UI polish,7November | Persistent pre/post controls, automatic numbering, shared theme/alignment, compact transport, no detached-player button. Coalesced seeks measured80pointer updates→20decoder seeks on a synthetic long-GOP clip; not real-footage latency. Evidence: `PlayChop-Experiments/studio-style-_buawzgi` |
| Export reload race,8November | Nested dialog event loop could read global padding while match reload was pending. Preserve game controls and captured preferences; deterministic native UI regression. Failed Actions42408776144 never established installed success |
| Model payload cleanup,8November | Decision-only catalogue; retired legacy PKL, S3D/MobileNet payloads.46inventoried files≈278MiB removed from isolated alternatives, with `approach-study-81sxifad/retired-models-2026-11-08.json` receipt. Transformer/approved weights retained |
| Windows catalogue hashes,9November | Git CRLF conversion changed immutable JSON bytes; `/models/** -text` and checkout regression preserve exact hashes. Minimal fixtures isolate production default discovery rather than downloading weights. Failed Actions42460587710/42460587695 are historical failures |
| Native setup shutdown | Own/wait for setup thread pool, destroy Qt objects explicitly and preserve fault logs; synthetic export alone did not establish safe natural shutdown |
| v1.0.1 publication | GitHub per-file2GiB limit blocked oversized runtime archive; do not claim it was a working published installer set |
| v1.0.2 | Public at `aaeda2a`; four installers plus runtime parts. All eight build/install jobs42895389027 passed, promotion42900356189 passed. First-release exception used because no upgrade baseline |
| v1.0.3 | Public at `ce53c48`; four installers only, upstream private engines. All eight build/install/upgrade jobs43484622175 passed; promotion43591331562 and tag tests43591331575 passed. All live upload digests matched installed-candidate evidence |
| Shared Python integrity | Second engine must verify existing shared Python before executing it; malformed receipts and POSIX permission damage must be repairable. Native smoke must retain trusted receipts until both engines are installed |
| Installer-only update metadata | Compact tested evidence lives in release notes and must match GitHub asset digests. Older Studio ≤1.0.2 needs one manual installer download. Do not silently weaken download validation |
| Hosted API limits | Authenticate the fixed upgrade-baseline API lookup with CI token; never forward it on redirects; pin downloaded installer hashes |
| Source-join research correction |50 ms clip padding could borrow a neighboring frame; strict source partitions reduce exploratory74.70→74.39%. Keep original and corrected evidence distinct |

## Documentation cleanup — 14 November

Replaced dated detector, pipeline, UI, installer, runtime, refiner, feature-pool
and neural progress reports with [current model status](model-status.md), the
[research ledger](research-ledger.md) and this engineering summary. Preserved the
compact machine-readable approach report under `research/`. Reduced the old
chronological model work log to an evaluation protocol, and replaced completed
roadmap phases with actual remaining work. No application code or user data was
changed. Git retains all removed narrative and numeric detail.

Historical numeric-runtime export validation (5November) used a 198-feature / 5 Hz
classifier and 24,968threshold/adjacent-float/NaN probes: maximum error1.11e-16.
Prototype Linux reference/inference-only installers measured 439.4/228.9 MiB
(extracted regular files 1753.8/807.2 MiB). These were temporary pre-release builds,
not today's bootstrap installer sizes or the current trained classifier.

Cleanup validation: 103 local Markdown links resolved; five packaging tests and
source/wheel build, allowlist and installed CLI smoke checks passed. The retained
historical JSON compares equal after parsing to its original Git version. The
`docs/` tree shrank from29to23files (about43%logical bytes). Application runtime,
trained weights, private raw experiment outputs and user data were preserved.
