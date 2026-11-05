"""Small frozen runtime identity, readable without importing inference engines."""
import json
from pathlib import Path
import sys


def profile():
    if not getattr(sys, 'frozen', False):
        return 'reference'
    path = Path(sys._MEIPASS) / 'runtime-profile.json'
    if not path.exists():
        return 'reference'  # older standalone builds
    if path.stat().st_size > 1024:
        raise ValueError('Invalid bundled runtime profile')
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or type(value.get('schema')) is not int or value['schema'] != 1
            or value.get('profile') not in ('reference', 'inference')):
        raise ValueError('Unsupported bundled runtime profile')
    return value['profile']
