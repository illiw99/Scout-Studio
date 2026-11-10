# Small installers and automatic setup

Scout Studio's native installers contain the review interface, media/export
support and our approved decision models. Large ML dependencies are separate
processing engines. The same first-launch setup wizard runs on Windows, macOS
and Linux before automatic play detection is used.

Setup checks the computer without importing Torch into the GUI. A working NVIDIA
driver selects the CUDA engine on Windows/Linux; other machines select the CPU
engine. macOS downloads the engine for its native architecture, with Metal used
when available. Every engine is checked after preparation. An unavailable CUDA
engine falls back to the CPU download. Unsupported AMD/Intel GPU acceleration
currently uses CPU processing, not CUDA libraries.

The wizard shows progress, supports cancellation/retry and can repair a damaged
downloaded engine. Skipping setup leaves saved reviews and clip export available
offline. Detection and provider-dependent model-pack operations reopen setup when
needed. Subsequent launches verify and reuse the cached engine. Setup never
installs packages into a system Python environment or changes graphics drivers.

Downloads come from this repository's matching release. Each part is at most
1 GiB and verified against the catalogue embedded in the installer. Extraction
rejects links/path traversal, checks the complete file inventory and hashes,
and publishes an engine atomically only after verification. Downloads and checks
run outside the Qt UI thread; incomplete engines are never launched.

Processing runs in an isolated subprocess using the existing job/event/result
protocol. CPU fallback, model selection, reviewed-play protection and cache
identities remain part of the existing pipeline. Our trained decision models
remain bundled; generic detector weights are fetched separately from their
pinned official source and CPU graphs are prepared once as needed.

Engine caches live below `Scout/engines` in the OS user cache (LocalAppData on
Windows, Library/Caches on macOS, XDG_CACHE_HOME or ~/.cache on Linux). They are
separate from footage, model packs, GT, fields and saved reviews. Temporary
archive parts are removed after preparation. Source/Python installations keep
their own explicitly installed dependencies; they do not use this frozen-app
setup flow.

## Build and release validation

Each native runner builds headless CPU/CUDA engines from separate exact/hash-pinned
environments (one native CPU/Metal wheel on each macOS architecture). Every engine
performs synthetic TorchVision, numeric classifier, ONNX and available OpenVINO
checks before packaging. It also probes every catalogue-pinned bundled pipeline
head before setup succeeds. The bootstrap GUI installs those trusted recipes as
verified data; arbitrary recipe imports run their compatibility probe in the
prepared worker. Engines contain no Qt. The small GUI excludes Torch,
TorchVision, Ultralytics, sklearn, matplotlib and ONNX/OpenVINO providers.

Installed smoke tests use the exact catalogue-pinned runtime files from the same
candidate, probe every native engine and run a real worker job, the setup wizard,
playback/export and uninstall checks. Promotion binds every runtime part to the
same build/test identities and copies the tested bytes unchanged. Both installers
and parts must fit GitHub's per-asset size limit before release promotion.

A CUDA-capable build/probe on a GPU-less runner does not establish real NVIDIA
hardware execution. Native Metal quality/performance and old-to-new upgrades need
separate evidence. Historical play-quality and Linux speed measurements describe
the previously evaluated CUDA/OpenVINO pipeline, not a new full-game evaluation
of every engine produced here. No model retraining or GT modification is part of
this packaging change. See `runtime-delivery-contract.md` for the developer schema.

## Version history

The `v1.0.1` tag was created after all native installation tests passed. Its release
promotion verified the candidates but GitHub rejected the Windows CUDA installer
for exceeding 2 GiB. That tag is preserved. Version 1.0.2 introduces this separate
engine delivery to solve the publication limit across all supported systems.
