"""Locating ffmpeg, probing files, streaming frames and writing clips.

The bundled ``imageio-ffmpeg`` binary is used when the system has no ffmpeg of
its own, so the pipeline works without root.  That build has no ffprobe and no
drawtext filter, which is why metadata is parsed from ``ffmpeg -i`` and overlays
are drawn with OpenCV instead.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import threading
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
from .processes import background_options

_DUR = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d\.\d+)")
_VID = re.compile(r"Stream #\d+:(\d+).*?: Video:.*?(\d{2,5})x(\d{2,5})")
_AUD = re.compile(r"Stream #\d+:(\d+).*?: Audio:")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_CREATED = re.compile(r"creation_time\s*:\s*(\S+)")


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Standalone apps use their bundled build; source installs prefer system."""
    if not getattr(sys, 'frozen', False):
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=1)
def has_filter(name: str = "drawtext") -> bool:
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-filters"],
        capture_output=True, text=True,
        **background_options(),
    ).stdout
    return bool(re.search(rf"^\s*\S+\s+{re.escape(name)}\s", out, re.M))


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    created: _dt.datetime | None
    #: Stream numbers within the file.  Not every camera puts video first - one
    #: writes audio as stream 0 - and anything that names streams by index has
    #: to be told which is which rather than assuming.
    video_index: int = 0
    audio_index: int | None = 1

    @property
    def name(self) -> str:
        return self.path.name


def probe(path: str | Path) -> MediaInfo:
    """Read duration/geometry/creation time by parsing ``ffmpeg -i``."""
    path = Path(path)
    txt = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
        **background_options(),
    ).stderr

    m = _DUR.search(txt)
    if not m:
        raise ValueError(f"{path.name}: no duration - unreadable or still copying")
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])

    v = _VID.search(txt)
    if not v:
        raise ValueError(f"{path.name}: no video stream")
    video_index = int(v[1])
    width, height = int(v[2]), int(v[3])
    a = _AUD.search(txt)
    audio_index = int(a[1]) if a else None

    line = txt[v.start(): txt.find("\n", v.start())]
    f = _FPS.search(line)
    fps = float(f[1]) if f else 30.0

    created = None
    c = _CREATED.search(txt)
    if c:
        try:
            created = _dt.datetime.fromisoformat(c[1].replace("Z", "+00:00"))
        except ValueError:
            pass

    return MediaInfo(path, duration, width, height, fps, created,
                     video_index, audio_index)


@lru_cache(maxsize=1)
def ffmpeg_has_cuda() -> bool:
    """Whether the selected FFmpeg was built with CUDA decoding support.

    The driver and input codec are checked by attempting the actual decode.
    """
    try:
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
            **background_options(),
        )
        return result.returncode == 0 and "cuda" in result.stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return False


def iter_frames(
    path: str | Path,
    fps: float,
    width: int,
    height: int,
    start: float = 0.0,
    duration: float | None = None,
    gray: bool = False,
    *,
    hwaccel: bool = True,
    gpu_resize: bool = False,
    cancel_event=None,
) -> Iterator[np.ndarray]:
    """Stream frames using CUDA when available, otherwise CPU.

    Optional GPU resizing avoids downloading source-resolution frames. Failed
    resizing retries CUDA decoding alone, then CPU, without repeating frames.
    Set ``hwaccel=False`` to force CPU.
    """
    delivered = 0
    if cancel_event is not None and cancel_event.is_set():
        return
    if hwaccel and ffmpeg_has_cuda():
        for resize in ([True,False] if gpu_resize else [False]):
            try:
                options=dict(cuda=True,cancel_event=cancel_event)
                if resize:options['gpu_resize']=True
                for i,frame in enumerate(_iter_frames(path,fps,width,height,start,duration,gray,**options)):
                    if i>=delivered:
                        delivered+=1
                        yield frame
                return
            except RuntimeError as exc:
                if cancel_event is not None and cancel_event.is_set():return
                message=(f'CUDA resize failed for {Path(path).name}; retrying without GPU resizing. {exc}' if resize else
                         f'CUDA decode failed for {Path(path).name}; using CPU. {exc}')
                warnings.warn(message,RuntimeWarning,stacklevel=2)
    for i, frame in enumerate(_iter_frames(path, fps, width, height, start, duration, gray,
                                          cancel_event=cancel_event)):
        if i >= delivered:
            yield frame


def _iter_frames(
    path: str | Path,
    fps: float,
    width: int,
    height: int,
    start: float = 0.0,
    duration: float | None = None,
    gray: bool = False,
    *,
    cuda: bool = False,
    gpu_resize: bool = False,
    cancel_event=None,
) -> Iterator[np.ndarray]:
    """Yield decoded frames as numpy arrays, resampled to ``fps``.

    Frames come over a pipe, which returns short reads on large buffers, so each
    frame is assembled with an explicit read loop.
    """
    pix = "gray" if gray else "rgb24"
    chans = 1 if gray else 3
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if cuda:
        cmd += ["-hwaccel", "cuda"]
        if gpu_resize:cmd += ['-hwaccel_output_format','cuda']
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    video_filter=(f'fps={fps},scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12,format={pix}'
                  if cuda and gpu_resize else f'fps={fps},scale={width}:{height}:flags=bilinear')
    cmd += [
        "-vf", video_filter,
        "-pix_fmt", pix, "-f", "rawvideo", "-",
    ]

    nbytes = width * height * chans
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=nbytes * 8,
                            **background_options())
    finished = threading.Event()
    watcher = None
    if cancel_event is not None:
        def cancel_decoder():
            while not finished.wait(.05):
                if cancel_event.is_set():
                    try:
                        proc.kill()  # Interrupt a blocking stdout.read, including on Windows.
                    except ProcessLookupError:
                        pass
                    return
        watcher = threading.Thread(target=cancel_decoder,daemon=True)
        watcher.start()
    drained = False
    try:
        while True:
            buf = bytearray()
            while len(buf) < nbytes:
                chunk = proc.stdout.read(nbytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < nbytes:
                drained = True
                break
            arr = np.frombuffer(bytes(buf), np.uint8)
            yield arr.reshape((height, width) if gray else (height, width, chans))
    finally:
        try:
            if not drained:
                # Caller stopped early: kill rather than leave a blocked pipe.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            if proc.stdout:
                proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            if proc.stderr:
                proc.stderr.close()
            rc = proc.wait()
            if drained and rc not in (0, None) and err.strip() and not (cancel_event is not None and cancel_event.is_set()):
                raise RuntimeError(f"ffmpeg decode failed for {Path(path).name}:\n{err[-1500:]}")
        finally:
            finished.set()
            if watcher is not None:
                watcher.join(timeout=1)


def grab_frame(path: str | Path, t: float, width: int = 0, height: int = 0, *,
               hwaccel: bool = True, gpu_resize: bool = False) -> np.ndarray:
    """Single RGB frame at time ``t`` (native size when width/height are 0)."""
    info = probe(path)
    w = width or info.width
    h = height or info.height
    options={}
    if not hwaccel:options['hwaccel']=False
    if gpu_resize:options['gpu_resize']=True
    for frame in iter_frames(path,fps=1000,width=w,height=h,start=t,duration=.2,**options):
        return frame
    raise ValueError(f"no frame at t={t} in {Path(path).name}")


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, **background_options())
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{res.stderr[-2000:]}")
