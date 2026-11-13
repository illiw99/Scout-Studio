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
            or value.get('profile') not in ('reference', 'inference', 'bootstrap')):
        raise ValueError('Unsupported bundled runtime profile')
    return value['profile']


def processing_identity():
    """Namespace new worker caches without changing older/source cache keys."""
    if not getattr(sys, 'frozen', False):
        value=getattr(sys,'_scout_engine_identity',None)
        if value is None:return None
    else:
        path = Path(sys._MEIPASS) / 'engine-id.json'
        if not path.is_file():return None
        if path.stat().st_size > 1024:raise ValueError('Invalid processing-engine identity')
        value = json.loads(path.read_text())
    from .processing_runtime import ENGINE_IDS
    if not isinstance(value, dict) or value.get('engine_id') not in ENGINE_IDS:
        raise ValueError('Unknown processing-engine identity')
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ('torch','torchvision','ultralytics','numpy','opencv-python-headless'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = 'unavailable'
    result=dict(engine=value['engine_id'], dependencies=versions)
    if 'requirements_sha256' in value:
        from .processing_runtime import _hash
        if not _hash(value.get('requirements_sha256')) or not _hash(value.get('wheel_sha256')):
            raise ValueError('Invalid pip processing identity')
        result['recipe']={key:value[key] for key in ('requirements_sha256','wheel_sha256')}
    return result
