# Installation

[README](../README.md) · [Development](development.md)

## Standalone installers

Download an installer from [GitHub Releases](https://github.com/illiw99/Scout-Studio/releases).
Installers appear after a version tag promotes the exact files from a successful
main-branch build and installation-test run; tags do not rebuild installers.
Python, the native Qt GUI, media/export support and our decision models are bundled.
On first launch the same setup wizard automatically downloads and verifies the
processing engine appropriate for the computer. See [automatic setup](runtime-delivery.md).

| Platform | Download | Install |
| --- | --- | --- |
| Windows x64 | `Scout-Studio-<version>-windows-x86_64.exe` | Run setup; open Scout Studio from Start. No administrator account is needed. |
| Ubuntu 22.04+ / compatible Debian x64 | `Scout-Studio-<version>-linux-x86_64.deb` | Run `sudo apt install ./Scout-Studio-<version>-linux-x86_64.deb`; open Scout Studio from Applications. |
| macOS Apple Silicon | `Scout-Studio-<version>-macos-arm64.dmg` | Open the disk image and drag Scout Studio to Applications. |
| macOS Intel | `Scout-Studio-<version>-macos-x86_64.dmg` | Open the disk image and drag Scout Studio to Applications. |

Windows and macOS installers currently have no publisher signing certificate or
Apple notarization. Their security systems may warn or block launch. Only allow
an installer you obtained from this repository's release and whose SHA-256 matches
`SHA256SUMS.txt`. Managed computers may require administrator approval. CI checks
installation and playback; it does not establish SmartScreen reputation or
Gatekeeper approval. macOS builds target the listed GitHub runner's OS generation;
older macOS versions are not verified.

Releases also include `release-manifest.json`, identifying the source commit,
build run/attempt, dependency versions and installed-app checks for each installer.
Checksums and recorded build/test identities do not replace publisher signatures.

New installations keep the game index in Qt's standard local application-data
directory, independently of the launch directory. Remembered libraries and
explicit `--workspace` selections remain supported. The Library menu reveals
the index folder, backs it up, and explicitly imports an existing index or backup.
Footage, fields, reviews, configuration
and exports live outside the installed application; uninstalling keeps that library.
Installers include our approved play decision model and selected Person
Transformer with their matching configuration. Large inference libraries are
downloaded by the first-launch setup wizard, rather than embedded in every
installer. Personal footage and generic
pretrained detector weights are excluded. First detection fetches verified YOLO
weights and, on CPU, prepares a cached OpenVINO/ONNX export. Later runs reuse
those cached files. Studio defaults to the approved automatic GPU/CPU pipeline;
explicit custom model selections remain supported. Existing review libraries can
be edited and exported without downloading a detector. See the
[model inventory](model-inventory.md) for weight origins and measured quality.

Setup downloads CUDA 12.8-capable PyTorch for supported NVIDIA systems on
Windows/Linux, otherwise the CPU engine. macOS receives its native CPU/Metal
engine. Earlier Windows
releases bundled CPU-only PyTorch and cannot use CUDA inference. AMD and Intel
GPUs are not selected by the current PyTorch backend. All installers bundle FFmpeg. CUDA inference and video decoding are
separate capabilities. A CUDA-capable detector does not enable FFmpeg GPU decoding.
See [detector performance](detection-performance.md) for faster model candidates.
For a custom system FFmpeg or Linux CUDA inference, use the Python installation
below. Remove the Linux package with `sudo apt remove scout-studio`, use Windows
Installed apps, or move the macOS application to Trash.

## Python installation

Python 3.10+ is required for a source installation. Run from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[reference]'
scout --help
```

Install Shotcut separately if you prefer it for reviewing. Scout uses system FFmpeg when available,
otherwise the binary supplied by `imageio-ffmpeg`. Detection runs on CPU or CUDA;
the first detection run may download YOLO weights. The versioned decision models
in `models/` are also included in wheels. The legacy `models/play.pkl` has been
retired; the approved default uses a numeric `classifier.json` with its matching
pipeline recipe. Use `--segmenter rules` to generate initial proposals without
the trained decision model.

### Optional NVIDIA video decoding (Ubuntu 24.04)

Install the system FFmpeg package and verify your NVIDIA driver:

```bash
sudo apt update
sudo apt install ffmpeg
nvidia-smi
ffmpeg -hide_banner -hwaccels
```

Look for `cuda` in the hardware acceleration list. The bundled imageio FFmpeg
may lack CUDA support; Scout prefers `ffmpeg` on your PATH. A CUDA-capable
build and working NVIDIA driver are needed; a prebuilt binary does not require
installing the full CUDA Toolkit. If your FFmpeg has no `cuda` entry, use a build
with NVIDIA decoding support (see [NVIDIA's FFmpeg guide](https://developer.nvidia.com/ffmpeg)).

Test an actual source file, since a listed accelerator alone does not prove
your driver and codec work:

```bash
ffmpeg -hide_banner -loglevel error -hwaccel cuda \
  -i media/W2_SAI-VET/DJI_0063.MP4 -t 3 -an -f null -
```

Exit status zero means this short decode succeeded. Restart Scout after
installation. No extra Scout flags are needed:

```bash
scout --match W2_SAI-VET run
```

Python checks the selected FFmpeg for CUDA support and attempts GPU decoding.
If CUDA is absent, it uses CPU directly. If GPU decoding fails, it warns and
retries on CPU, skipping frames already delivered. Cached detections remain
valid. This does not change YOLO's separate automatic inference device choice.

Check the binary and build support from Python in the activated virtualenv:

```bash
python -c 'from scout.media import ffmpeg_exe, ffmpeg_has_cuda; print(ffmpeg_exe()); print("CUDA build support:", ffmpeg_has_cuda())'
```

Frames still pass through CPU memory for preprocessing. GPU decoding can reduce
CPU load, but the speedup depends on the footage and hardware. Python callers
can compare against `iter_frames(..., hwaccel=False)` to force CPU decoding.
See [FFmpeg's hardware acceleration documentation](https://ffmpeg.org/ffmpeg.html).


## Desktop application

Install the optional desktop dependency in the same environment:

```bash
python -m pip install -e '.[desktop,reference]'
python -m scout.desktop
```

`scout-studio` is the installed entry point. Linux source checkouts can add an
Applications-menu entry with `python scripts/install_studio.py`. The installer
can provision the small X11 cursor dependency locally when apt tools are
available. Windows and macOS can use the Python entry point; native display
behavior on those platforms has not been checked by the Linux test suite.

## Installing a built wheel

After [building the packages](development.md#build-packages), install the wheel
into the target environment, adding `[desktop]` for the native app:

```bash
python -m pip install './dist/scout_flag-1.0.0-py3-none-any.whl[desktop]'
scout-studio
```

The wheel is an installable Python package, not a standalone executable or a
published PyPI release. Models, detector weights, and match data are separate.
A wheel installation can review and export existing libraries; model-based
detection requires a classifier configured through `model_path` in the library's
`scout.json`. Source checkouts include `models/play.pkl`; it is deliberately
excluded from redistributable archives. The rules segmenter is available without
a trained play classifier.

For approved offline default models and matching standalone media engines, see
[default-model deployment](default-model.md). Studio chooses the active/bundled
profile automatically while preserving explicit saved profile choices.
