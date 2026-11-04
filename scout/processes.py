"""Platform options for background helpers that still expose output pipes."""
from __future__ import annotations

import sys


def background_options() -> dict[str, int]:
    """Prevent Windows console creation without discarding logs or detaching.

    CREATE_NO_WINDOW is a process creation flag, so the window never flashes.
    Keep ordinary process handles, pipes and cancellation behavior intact.
    """
    if sys.platform == 'win32':
        return {'creationflags': 0x08000000}  # subprocess.CREATE_NO_WINDOW
    return {}
