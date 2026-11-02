"""An incomplete installed-app smoke report must never pass the release gate."""
import importlib.util
from pathlib import Path
import unittest


def checker():
    script = Path(__file__).resolve().parents[1] / 'scripts/test-installer.py'
    spec = importlib.util.spec_from_file_location('installer_checks', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallerValidationTests(unittest.TestCase):
    def test_rejects_source_execution_wrong_version_and_partial_success(self):
        validate = checker().validate_report
        checks = dict.fromkeys(('dependencies', 'ffmpeg', 'export', 'gui', 'playback', 'cli'), True)
        report = dict(ok=True, version='1.0.0', frozen=True, checks=checks)
        validate(report, '1.0.0')
        for replacement in ({'ok': False}, {'frozen': False}, {'version': '0.9.0'},
                            {'checks': {'gui': True}}):
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                validate({**report, **replacement}, '1.0.0')


if __name__ == '__main__':
    unittest.main()
