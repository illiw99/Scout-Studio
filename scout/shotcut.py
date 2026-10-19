"""Shotcut / MLT project export.

Two projects are written.  ``plays.mlt`` puts just the detected plays on the
timeline, back to back, ready to watch through or export.  ``review.mlt`` puts
the untouched recording on the timeline with a marker over every play, which is
the one to open when checking whether a boundary is in the right place.

Times are written in MLT's clock format.  ``out`` points at the last frame that
is included, not one past it, so it is a frame short of the end.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from .config import Config
from .segment import Play
from .timeline import Clip

_MARKER_COLOURS = ["#ff8a2be2", "#ff1e90ff", "#ff2ca02c", "#ffd2691e"]


def clock(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _profile(clip: Clip) -> str:
    fps = clip.info.fps
    # NTSC rates are n*1000/1001 and must be written exactly, or Shotcut drifts
    # against the source over a long timeline.
    ntsc = {23.98: (24000, 1001), 29.97: (30000, 1001),
            47.95: (48000, 1001), 59.94: (60000, 1001)}
    num = den = None
    for r, (n, d) in ntsc.items():
        if abs(fps - r) < 0.06:
            num, den = n, d
            break
    if num is None:
        num, den = int(round(fps * 1000)), 1000
    return (f'  <profile description="automatic" width="{clip.info.width}" '
            f'height="{clip.info.height}" progressive="1" sample_aspect_num="1" '
            f'sample_aspect_den="1" display_aspect_num="16" display_aspect_den="9" '
            f'frame_rate_num="{num}" frame_rate_den="{den}" colorspace="709"/>')


def _producer(pid: str, clip: Clip, tin: float, tout: float, caption: str) -> str:
    # The stream numbers come from the file itself.  "novalidate" means MLT
    # believes what it is told, so getting these wrong shows the clip on the
    # timeline with no picture.
    audio = clip.info.audio_index
    audio_prop = (f'\n    <property name="audio_index">{audio}</property>'
                  if audio is not None else
                  '\n    <property name="audio_index">-1</property>')
    return f"""  <producer id="{pid}" in="{clock(tin)}" out="{clock(tout)}">
    <property name="length">{clock(clip.info.duration)}</property>
    <property name="eof">pause</property>
    <property name="resource">{escape(str(clip.path.resolve()))}</property>
    <property name="mlt_service">avformat-novalidate</property>
    <property name="seekable">1</property>{audio_prop}
    <property name="video_index">{clip.info.video_index}</property>
    <property name="mute_on_pause">0</property>
    <property name="shotcut:caption">{escape(caption)}</property>
  </producer>"""


def _tractor(total: float, markers: str = "") -> str:
    return f"""  <tractor id="tractor0" title="Scout" global_feed="1" in="{clock(0)}" out="{clock(total)}">
    <property name="shotcut">1</property>
    <property name="shotcut:projectAudioChannels">2</property>
    <property name="shotcut:projectFolder">0</property>
{markers}    <track producer="background"/>
    <track producer="playlist0"/>
    <transition id="transition0">
      <property name="a_track">0</property>
      <property name="b_track">1</property>
      <property name="mlt_service">mix</property>
      <property name="always_active">1</property>
      <property name="sum">1</property>
    </transition>
    <transition id="transition1">
      <property name="a_track">0</property>
      <property name="b_track">1</property>
      <property name="version">0.1</property>
      <property name="mlt_service">frei0r.cairoblend</property>
      <property name="threads">0</property>
      <property name="disable">1</property>
    </transition>
  </tractor>"""


def _document(clip0: Clip, total: float, producers: list[str], entries: list[str],
              markers: str = "") -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<mlt LC_NUMERIC="C" version="7.0.0" title="Scout" producer="main_bin">
{_profile(clip0)}
  <playlist id="main_bin">
    <property name="xml_retain">1</property>
  </playlist>
  <producer id="black" in="{clock(0)}" out="{clock(total)}">
    <property name="length">{clock(total + 1)}</property>
    <property name="eof">pause</property>
    <property name="resource">0</property>
    <property name="aspect_ratio">1</property>
    <property name="mlt_service">color</property>
    <property name="mlt_image_format">rgba</property>
    <property name="set.test_audio">0</property>
  </producer>
  <playlist id="background">
    <entry producer="black" in="{clock(0)}" out="{clock(total)}"/>
  </playlist>
{chr(10).join(producers)}
  <playlist id="playlist0">
    <property name="shotcut:video">1</property>
    <property name="shotcut:name">V1</property>
{chr(10).join(entries)}
  </playlist>
{_tractor(total, markers)}
</mlt>
"""


def _segments(clips: list[Clip], start: float, end: float):
    """Split a global span into (clip, local_in, local_out) pieces."""
    for c in clips:
        if c.end <= start or c.start >= end:
            continue
        s = max(start, c.start) - c.start
        e = min(end, c.end) - c.start
        if e - s > 0.04:
            yield c, s, e


def write_plays(cfg: Config, clips: list[Clip], plays: list[Play],
                dst: Path) -> Path:
    """Timeline containing only the plays, in order."""
    frame = 1.0 / (clips[0].info.fps or 30.0)
    producers, entries = [], []
    total = 0.0
    n = 0
    for p in plays:
        for c, s, e in _segments(clips, p.start, p.end):
            pid = f"producer{n}"
            producers.append(_producer(pid, c, s, e - frame, f"play {p.index:03d}"))
            entries.append(f'    <entry producer="{pid}" in="{clock(s)}" '
                           f'out="{clock(e - frame)}"/>')
            total += e - s
            n += 1
    if not entries:
        raise SystemExit("no plays to export")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_document(clips[0], total - frame, producers, entries))
    return dst


def write_review(cfg: Config, clips: list[Clip], plays: list[Play],
                 dst: Path) -> Path:
    """The full recording with a marker over every detected play.

    Markers span the clip that would be cut, including its padding, so editing a
    marker here and re-cutting gives exactly what the marker showed.
    """
    frame = 1.0 / (clips[0].info.fps or 30.0)
    producers, entries = [], []
    for i, c in enumerate(clips):
        pid = f"producer{i}"
        producers.append(_producer(pid, c, 0.0, c.info.duration - frame, c.path.name))
        entries.append(f'    <entry producer="{pid}" in="{clock(0)}" '
                       f'out="{clock(c.info.duration - frame)}"/>')
    rows = []
    for i, p in enumerate(plays):
        rows.append(f"""      <properties name="{i}">
        <property name="text">Play {p.index}</property>
        <property name="start">{clock(p.start)}</property>
        <property name="end">{clock(p.end)}</property>
        <property name="color">{_MARKER_COLOURS[i % len(_MARKER_COLOURS)]}</property>
      </properties>""")
    markers = ('    <properties name="shotcut:markers">\n'
               + "\n".join(rows) + "\n    </properties>\n") if rows else ""
    total = clips[-1].end - frame
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(".mlt.tmp")
    temporary.write_text(_document(clips[0], total, producers, entries, markers))
    temporary.replace(dst)
    return dst
