"""Readable, deterministic labels on the full match clock."""
import re

_GENERATED = re.compile(r'^Play(?:\s+\d+|\s+\d{2,}:\d{2}:\d{2}\.\d{3})?$', re.IGNORECASE)


def time_label(seconds):
    whole, milliseconds = divmod(max(0, round(seconds*1000)), 1000)
    return f'Play {whole//3600:02d}:{whole//60%60:02d}:{whole%60:02d}.{milliseconds:03d}'


def label_for(text, start):
    """Keep custom coaching labels; update generated labels when ranges move."""
    return time_label(start) if not text.strip() or _GENERATED.fullmatch(text.strip()) else text
