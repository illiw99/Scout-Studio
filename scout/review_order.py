"""Explicit, conflict-safe camera reordering without changing reviewed footage."""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from . import groundtruth
from .shotcut import clock
from .timeline import filename_key


def order_by_filename(path: Path, expected_hash: str) -> dict:
    """Reorder full-source entries, mapping range and point markers with them.

    Marker pieces that remain adjacent are merged. A range whose source pieces
    become separated is split, preserving annotations and all marker properties.
    Other XML is retained. Footage and detection caches are never written.
    """
    path = Path(path).resolve()
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != expected_hash:
        raise ValueError('This review changed. Refresh it before ordering camera files.')
    groundtruth.load(path)
    _, fps = groundtruth._read(path)
    root = ET.fromstring(original, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    playlist = groundtruth._playlist(root)
    clips = groundtruth.project_clips(path)
    entries = playlist.findall('entry')
    durations = [groundtruth.parse_clock(e.get('out', '0'), fps)
                 - groundtruth.parse_clock(e.get('in', '0'), fps)+1/fps for e in entries]
    ends = [clip.start+duration for clip, duration in zip(clips, durations)]
    order = sorted(range(len(clips)), key=lambda i: filename_key(clips[i].path))
    if order == list(range(len(clips))):
        return dict(changed=False, files=[c.path.name for c in clips], backup=None, split_ranges=0)
    # Only the original single full-footage track is supported. Global effects
    # or other timed tracks cannot safely retain their meaning after reordering.
    tractors = root.findall('tractor')
    global_effects = (root.findall('filter') + root.findall('playlist/filter')
                      + root.findall('tractor/filter'))
    timed_transitions = [t for t in root.findall('tractor/transition')
                         if t.get('in') is not None or t.get('out') is not None]
    if len(tractors) != 1 or global_effects or timed_transitions:
        raise ValueError('Ordering requires the original review without global effects.')
    if any(t.get('producer') not in ('background', playlist.get('id'))
           for t in tractors[0].findall('track')):
        raise ValueError('Ordering additional timeline tracks is unsupported.')
    starts = {}
    total = 0.
    for index in order:
        starts[index] = total
        total += durations[index]
    for entry in entries:
        playlist.remove(entry)
    for index in order:
        playlist.append(entries[index])

    def point(seconds):
        for index, clip in enumerate(clips):
            if seconds < ends[index]-1e-7 or index == len(clips)-1:
                return starts[index]+seconds-clip.start
        raise ValueError('Marker outside the review timeline.')

    block = root.find(".//properties[@name='shotcut:markers']")
    split_ranges = 0
    if block is not None:
        rows = []
        for node in list(block):
            properties = {p.get('name'): p for p in node if p.tag == 'property'}
            if 'start' not in properties:
                rows.append((math.inf, node))
                continue
            start = groundtruth.parse_clock(properties['start'].text or '0', fps)
            end = groundtruth.parse_clock(properties['end'].text or '0', fps) if 'end' in properties else start
            if start == end:
                spans = [(point(start), point(start))]
            else:
                spans = sorted((starts[i]+max(start, clip.start)-clip.start,
                                starts[i]+min(end, ends[i])-clip.start)
                               for i, clip in enumerate(clips) if start < ends[i] and end > clip.start)
                merged = []
                for a, b in spans:
                    if merged and abs(merged[-1][1]-a) < 1e-6:
                        merged[-1] = (merged[-1][0], b)
                    else:
                        merged.append((a, b))
                spans = merged
                split_ranges += max(0, len(spans)-1)
            for a, b in spans:
                mapped = copy.deepcopy(node)
                mapped.find("property[@name='start']").text = clock(a)
                if 'end' in properties:
                    mapped.find("property[@name='end']").text = clock(b)
                rows.append((a, mapped))
        for node in list(block):
            block.remove(node)
        for index, (_, node) in enumerate(sorted(rows, key=lambda item: item[0])):
            node.set('name', str(index))
            block.append(node)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.mlt')
    temp = Path(temporary)
    backup = path.with_name(f'{path.stem}.before-source-order-{digest[:12]}.mlt')
    try:
        with os.fdopen(fd, 'wb') as handle:
            ET.ElementTree(root).write(handle, encoding='utf-8', xml_declaration=True)
            handle.flush()
            os.fsync(handle.fileno())
        groundtruth.load(temp)
        resulting = groundtruth.project_clips(temp)
        if [c.path for c in resulting] != [clips[i].path for i in order]:
            raise ValueError('Reordered review failed source validation.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('This review changed while ordering camera files. Refresh it.')
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError('An unrelated backup already uses this name.')
        else:
            with backup.open('xb') as handle:
                handle.write(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('This review changed while backing it up. Refresh it.')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return dict(changed=True, files=[c.path.name for c in resulting],
                backup=backup, split_ranges=split_ranges)
