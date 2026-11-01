"""Native Scout desktop application (optional PySide6 dependency)."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Launch Studio, explaining how to install the optional Qt dependency."""
    try:
        from .app import main as run
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6" or (exc.name or "").startswith("PySide6."):
            import sys

            print(
                'Scout Studio requires the desktop extra. Install it with:\n'
                '  python -m pip install "scout-flag[desktop]"\n'
                'From a source checkout, use: python -m pip install ".[desktop]"',
                file=sys.stderr,
            )
            return 1
        raise
    return run(argv)
