#!/usr/bin/env python3
"""Standalone entry point; also available as scout --match MATCH rename-clips."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scout.rename_clips import main

if __name__ == '__main__':
    raise SystemExit(main())
