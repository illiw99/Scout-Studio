# Standalone dependency snapshots

These six full transitive, hash-pinned snapshots target CPython 3.12. They include
the application, desktop and inference extras, plus PyInstaller and build tools.
They do not change the Python 3.10+ source-install contract or make desktop
dependencies mandatory for CLI users.

| File | Target | Torch |
| --- | --- | --- |
| `linux-x86_64-cpu.txt` | Linux x86_64, glibc 2.35+ (Ubuntu 22.04 baseline) | 2.7.1 CPU |
| `linux-x86_64-cu128.txt` | Linux x86_64, glibc 2.35+ | 2.7.1 CUDA 12.8, CPU fallback |
| `windows-x86_64-cpu.txt` | Windows x86_64 | 2.7.1 CPU |
| `windows-x86_64-cu128.txt` | Windows x86_64 | 2.7.1 CUDA 12.8, also runs on CPU |
| `macos-arm64.txt` | macOS 15 arm64 | 2.7.1 |
| `macos-x86_64.txt` | macOS 15 Intel | 2.2.2, torchvision 0.17.2, NumPy 1.26.4 |

Windows and Linux each have CPU-only and CUDA-capable snapshots. The CUDA
worker also supports CPU probing on machines without an NVIDIA GPU; first-launch
setup can choose the smaller CPU engine independently.
Intel macOS retains the last Intel Torch release and NumPy's compatible pre-2
ABI. Its older OpenCV and ONNX Runtime versions are selected by wheel/ABI
availability. All other dependencies resolve against the recorded cutoff.
Ultralytics 8.4.183 and scikit-learn 1.9.1 match the project's validated baseline.
OpenVINO is included on Windows/Linux only. Ultralytics requires both OpenCV
distributions and the pure-Python `nvidia-ml-py` monitoring package, including on
macOS; that package does not imply a CUDA runtime.

Create a **new** Python 3.12 virtual environment on the matching native platform:

```sh
python -m pip install --require-hashes --only-binary=:all: -r packaging/locks/<target>.txt
python -m pip install --no-deps --no-build-isolation '.[desktop,inference]'
python -m pip check
python scripts/dependency-locks.py --check
```

CI uses `scripts/dependency-locks.py --install --target <target>` inside a new
native Python3.12 virtual environment. It installs/verifies the full lock, then
removes both overlapping OpenCV distributions and reinstalls the **same hashed
headless wheel**. This avoids packaging OpenCV's second Qt distribution.
`--check-installed --target <target>` verifies the installed pins and allows
only Ultralytics' known missing `opencv-python` metadata requirement, satisfied
functionally by the pinned headless distribution. Other dependency errors fail.
The helper rejects populated or system-sharing virtual environments before any
pip action, ignores pip/user Python configuration and directs installation to
that venv's prefix. It installs the project with dependency resolution and build
isolation disabled through the same isolated pip invocation.

The first command must install the lock as requirements, not merely constraints.
The second must disable dependency resolution and build isolation, preventing
unlocked runtime or build dependencies from entering the candidate environment.
Do not add unconstrained `pip install --upgrade` steps afterward. The existing
installer build and native installed-artifact tests follow these commands.

Regeneration uses uv 0.9.7, official PyPI/PyTorch indexes, target-aware marker
resolution and binary-only candidates. In a fresh temporary tools environment:

```sh
python -m pip install uv==0.9.7
python scripts/dependency-locks.py --generate --uv /path/to/tools/bin/uv
python scripts/dependency-locks.py --verify-resolution --uv /path/to/tools/bin/uv
python -m unittest discover -s tests -p 'test_dependency_locks.py' -v
```

Use `--target <name>` to operate on one snapshot. Change the explicit baseline
and cutoff in `scripts/dependency-locks.py` deliberately, regenerate all locks,
and validate all four native builds before promoting a candidate. The helper
uses an isolated temporary cache and ignores uv/pip environment configuration.
macOS resolution sets `MACOSX_DEPLOYMENT_TARGET=15.0`; Linux targets glibc 2.35.
`--check` is offline structural validation; `--verify-resolution` independently
resolves and compares the complete result. Neither establishes successful native
installation, Torch operators, playback, codecs, GPU availability or packaging.
Native candidate CI must install and exercise each platform separately.

Processing packs use separate native CPU/CUDA build environments on Linux and
Windows. `scripts/build-runtime.py` checks every installed pin before freezing;
CUDA workers must pass a real CPU TorchVision NMS probe on GPU-less runners.
macOS uses one native CPU/MPS engine per architecture. Bootstrap installers are
frozen with `--runtime-profile bootstrap --runtime-files DIRECTORY`; they bundle
the merged, platform-only catalogue, native Qt and approved project model files,
while excluding ML frameworks. Pack archives materialize library links and split
into parts of at most 1 GiB. Reference/inference installer profiles remain usable.
