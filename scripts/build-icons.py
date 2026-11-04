#!/usr/bin/env python3
"""Compile the existing SVG logo into a Windows ICO; no artwork changes."""
from __future__ import annotations

from pathlib import Path
import hashlib
import struct


def build_icon(source: Path, target: Path) -> None:
    from PySide6.QtCore import QByteArray, QBuffer, QIODevice
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise ValueError(f'Invalid SVG logo: {source}')
    frames = []
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for size in (16, 24, 32, 48, 64, 128, 256):
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(0)
        image.setText('source-svg-sha256', source_hash)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not image.save(buffer, 'PNG'):
            raise RuntimeError(f'Cannot render {size}px icon')
        frames.append((size, bytes(data)))
    offset = 6 + 16 * len(frames)
    entries = []
    for size, frame in frames:
        entries.append(struct.pack('<BBBBHHII', size % 256, size % 256, 0, 0,
                                   1, 32, len(frame), offset))
        offset += len(frame)
    target.write_bytes(struct.pack('<HHH', 0, 1, len(frames)) + b''.join(entries)
                       + b''.join(frame for _, frame in frames))


if __name__ == '__main__':
    assets = Path(__file__).resolve().parents[1] / 'scout/desktop'
    build_icon(assets / 'scout.svg', assets / 'scout.ico')
