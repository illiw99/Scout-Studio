"""Read reviewed Shotcut range markers and validate their original source timeline."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


def parse_clock(s: str, fps: float = 1.0) -> float:
    """MLT clock time (``HH:MM:SS.mmm``) or a frame count, in seconds."""
    s = s.strip()
    if ":" not in s:
        return float(s) / fps
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    else:
        h, (m, sec) = 0, parts
    return int(h) * 3600 + int(m) * 60 + float(sec)


@dataclass
class Marked:
    index: int
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _read(path):
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: invalid MLT XML: {exc}") from exc
    profile = root.find("profile")
    if profile is None:
        raise ValueError(f"{path}: missing MLT profile")
    denominator = float(profile.get("frame_rate_den", "1"))
    if denominator <= 0:
        raise ValueError(f"{path}: invalid frame-rate denominator")
    fps = float(profile.get("frame_rate_num", "0")) / denominator
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid frame rate")
    return root, fps


def _playlist(root):
    playlists = [p for p in root.findall("playlist")
                 if p.find("property[@name='shotcut:video']") is not None]
    if not playlists:
        playlists = root.findall("playlist[@id='playlist0']")
    if len(playlists) != 1:
        raise ValueError("review must contain exactly one video track; edit markers on the original timeline")
    pl = playlists[0]
    if pl.find("blank") is not None:
        raise ValueError("review timeline contains gaps; restore the original footage timeline")
    return pl


def load(path: str | Path) -> list[Marked]:
    root, fps = _read(path)
    entries = _playlist(root).findall("entry")
    total = sum(parse_clock(e.get("out", "0"), fps)
                - parse_clock(e.get("in", "0"), fps) + 1/fps for e in entries)
    block = root.find(".//properties[@name='shotcut:markers']")
    rows = []
    if block is not None:
        for m in block:
            d = {p.get("name"): (p.text or "") for p in m}
            if "start" not in d:
                continue
            a = parse_clock(d["start"], fps)
            b = parse_clock(d.get("end", d["start"]), fps)
            if not all(math.isfinite(t) for t in (a,b)) or a < 0 or b < a or b > total + .002:
                raise ValueError(f"{path}: marker {d.get('text', '')!r} is outside the timeline or reversed")
            if b == a:  # Notes/point markers are not play ranges.
                continue
            rows.append((a, b, d.get("text", "")))
    rows.sort()
    return [Marked(i, a, b, txt) for i, (a, b, txt) in enumerate(rows, 1)]


def project_clips(path: str | Path):
    """Resolve full source videos in MLT timeline order, never directory order.

    Scout's review contract is to edit range markers, leaving footage intact.
    Reject unsupported edits instead of silently exporting the wrong frames.
    """
    from .media import probe
    from .timeline import Clip
    path = Path(path).resolve()
    root, fps = _read(path)
    producers = {p.get("id"): p for p in root if p.tag in ("producer", "chain")}
    base = Path(root.get("root", str(path.parent)))
    if not base.is_absolute():
        base = path.parent / base
    clips, seen, t = [], set(), 0.
    for entry in _playlist(root).findall("entry"):
        producer = producers.get(entry.get("producer"))
        if producer is None:
            raise ValueError("unsupported nested timeline or missing source producer")
        props = {p.get("name"): p.text or "" for p in producer.findall("property")}
        if "timewarp" in props.get("mlt_service", ""):
            raise ValueError("speed changes are unsupported; restore the original footage timeline")
        resource = props.get("resource", "")
        if not resource:
            raise ValueError("MLT source has no resource path")
        source = Path(resource)
        source = (base / source).resolve() if not source.is_absolute() else source.resolve()
        if source in seen:
            raise ValueError("repeated or split source videos are unsupported; edit review markers only")
        seen.add(source)
        info = probe(source)
        start = parse_clock(entry.get("in", "0"), fps)
        duration = parse_clock(entry.get("out", "0"), fps) - start + 1/fps
        if abs(start) > .002 or abs(duration - info.duration) > max(.02, 1/fps):
            raise ValueError(f"{source.name}: trimmed footage is unsupported; edit review markers only")
        clips.append(Clip(info, t, len(clips)))
        t += duration
    if not clips:
        raise ValueError("MLT review has no source videos")
    return clips


def to_plays(marks: list[Marked], pre: float = 0.0, post: float = 0.0):
    """Convert marks into Play objects so they can be cut or filmstripped."""
    from .segment import Play
    out = []
    for m in marks:
        out.append(Play(
            index=m.index, start=max(0.0, m.start - pre), end=m.end + post,
            snap=m.start, whistle=m.end, duration=m.duration,
            peak_action=0.0, peak_sprinters=0, median_players=0,
        ))
    return out


def renumber(path: str | Path, prefix: str = "Play", backup: bool = True) -> int:
    """Rename every marker to ``<prefix> N`` in chronological order.

    Markers added by hand in Shotcut are called "Marker 3", "Marker 15" and so
    on, numbered in the order they happened to be created rather than the order
    they occur, which makes the exported clips hard to line up against the
    timeline.  This renames them in time order and writes the project back.
    """
    path = Path(path)
    root, fps = _read(path)
    block = root.find(".//properties[@name='shotcut:markers']")
    if block is None:
        return 0

    entries = []
    for node in list(block):
        props = {p.get("name"): p for p in node}
        if "start" not in props:
            continue
        entries.append((parse_clock(props["start"].text or "0", fps), node, props))
    entries.sort(key=lambda e: e[0])

    for i, (_, node, props) in enumerate(entries, 1):
        label = f"{prefix} {i}"
        if "text" in props:
            props["text"].text = label
        else:
            el = ET.SubElement(node, "property")
            el.set("name", "text")
            el.text = label
        # Shotcut keys markers by their child name; keep that in time order too.
        node.set("name", str(i - 1))

    # Re-insert in the sorted order so the file reads the way it plays.
    for node in list(block):
        block.remove(node)
    for _, node, _ in entries:
        block.append(node)

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(path.read_bytes())
    xml = ET.tostring(root, encoding="unicode")
    path.write_text('<?xml version="1.0" standalone="no"?>\n' + xml)
    return len(entries)


def media_for(path: str | Path) -> str | None:
    """Which media file a Shotcut project points at.

    Lets a ground-truth project be handed over on its own: the file it grades is
    read out of the project rather than named again on the command line.
    """
    root = ET.parse(str(path)).getroot()
    for prod in root.iter("producer"):
        for prop in prod:
            if prop.get("name") == "resource" and (prop.text or "").strip() not in ("", "0"):
                return Path(prop.text.strip()).name
    for chain in root.iter("chain"):
        for prop in chain:
            if prop.get("name") == "resource" and (prop.text or "").strip() not in ("", "0"):
                return Path(prop.text.strip()).name
    return None
