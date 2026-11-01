"""Filename patterns shared by clip export and its native dialog."""
from __future__ import annotations


def pattern_names(count: int, pattern: str, start: int = 1,
                  pad: int | None = None) -> list[str]:
    """Expand one freely positioned ``<N>`` token into safe MP4 filenames.

    Automatic padding includes both the clip count and the final number.
    Zero is a valid start or explicit width; widths never truncate numbers.
    """
    for label, value in (("count", count), ("start", start), ("pad", pad)):
        if label == "pad" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if not isinstance(pattern, str) or pattern.count("<N>") != 1:
        raise ValueError("name pattern must contain exactly one <N>")
    # Portable filename rules also keep paths and control characters out of
    # manifests. The token itself is the only permitted use of angle brackets.
    literal = pattern.replace("<N>", "")
    if any(ord(c) < 32 or ord(c) == 127 or c in '/\\<>:"|?*' for c in literal):
        raise ValueError("name pattern must be a filename without separators or unsafe characters")
    if literal.endswith((" ", ".")):
        raise ValueError("name pattern must not end in a space or dot")
    if pad is None:
        pad = len(str(max(count, start + count - 1)))
    if pad > 251:
        raise ValueError("number width produces a filename longer than 255 bytes")
    names = []
    for number in range(start, start + count):
        name = pattern.replace("<N>", str(number).zfill(pad))
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        if len(name.encode("utf-8")) > 255:
            raise ValueError("name pattern produces a filename longer than 255 bytes")
        stem = name.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or stem in {
                f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}:
            raise ValueError("name pattern produces a reserved filename")
        names.append(name)
    if len({name.casefold() for name in names}) != count:
        raise ValueError("name pattern must produce unique filenames")
    return names
