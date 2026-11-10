# Processing engine delivery contract

The application is version 1.0.2. Frozen `bootstrap` installers include native Qt,
media/review/export dependencies, project decision models and a platform-specific
`runtime-catalog.json`. They exclude Torch, TorchVision, Ultralytics, sklearn,
matplotlib, ONNX, ONNX Runtime and OpenVINO. First-launch setup selects and prepares
a headless processing worker in the user's cache; source installations keep their
existing environment and reference/inference profiles remain supported.

Bundled catalogue schema (no arbitrary download URLs):

```json
{"schema":1,"application_version":"1.0.2","runtimes":{
 "linux-x86_64-cpu":{
  "entrypoint":"Scout-Worker/Scout-Worker",
  "archive":"tar.gz","uncompressed_size":123,
  "members":{"Scout-Worker/Scout-Worker":{"size":123,"sha256":"...","mode":493}},
  "parts":[{"name":"Scout-Runtime-1.0.2-linux-x86_64-cpu.tar.gz.part001","size":123,"sha256":"..."}]
 }}}
```

Engine ids: `windows-x86_64-cpu`, `windows-x86_64-cuda`, `linux-x86_64-cpu`,
`linux-x86_64-cuda`, `macos-arm64`, `macos-x86_64`. macOS Torch wheels include
CPU/MPS in one native engine. CPU/CUDA engines must use separately pinned native
environments. Catalogue for each GUI contains its platform's engine(s) only.

Pack archives contain a frozen `Scout-Worker` onedir, with every link materialized
as a regular file at build time and bounded, exact member SHA-256 identities.
Archive bytes are split into parts no larger than 1 GiB; every published asset
must be strictly below GitHub's 2 GiB limit. URLs derive exclusively from
`https://github.com/illiw99/Scout-Studio/releases/download/v<application_version>/<part.name>`.
Clients verify parts, each extracted member and the complete set before atomic
publication; no archive links or path traversal are accepted. Executable modes
come from trusted member metadata. Corrupt or partial engines never run.

Worker command contract: `Scout-Worker[.exe] --worker-job DESCRIPTOR`;
`--scout-cli ARGS...`; `--engine-probe REPORT.json`. The probe runs without Qt,
verifies TorchVision NMS, Torch/Ultralytics imports and reports available devices
and versions. It must support CPU probing in CUDA builds on GPU-less CI runners.
All normal processing stays within existing versioned worker job/event/result
contracts. GUI/export commands can still use the bundled small worker.

Build metadata: `build-info.json.processing_catalog` is `runtime-catalog.json`.
Candidate files include that catalogue and every referenced runtime part, plus
existing installer/helper/evidence files. Promotion verifies/copies every runtime
part unchanged and includes each in SHA256SUMS.txt, preserving build/test identity.
Installer tests preseed **only** the exact expected parts via explicit
`--runtime-files DIRECTORY` smoke-test arguments; local test files use the same
catalogue identities, never alternate hashes or arbitrary production URLs.
Fresh native smoke must prepare/probe each engine, execute a real worker command,
and then exercise the installed bootstrap GUI/media/export paths. Local synthetic
unit fixtures cover interrupted, corrupted, unsafe and incompatible downloads.
