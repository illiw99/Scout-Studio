"""An incomplete installed-app smoke report must never pass the release gate."""
import importlib.util
import contextlib
import ctypes
import io
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch


def checker():
    script = Path(__file__).resolve().parents[1] / 'scripts/test-installer.py'
    spec = importlib.util.spec_from_file_location('installer_checks', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallerValidationTests(unittest.TestCase):
    def test_embedded_windows_logo_is_verified_and_resource_handle_is_closed(self):
        module = checker()
        icon = Path(__file__).resolve().parents[1] / 'scout/desktop/scout.ico'
        data = icon.read_bytes()
        count = struct.unpack_from('<H', data, 4)[0]
        entries, resources = [], {}
        for index in range(count):
            row = struct.unpack_from('<BBBBHHII', data, 6 + index * 16)
            frame = data[row[-1]:row[-1] + row[-2]]
            resources[(3 << 16) | (index + 1)] = frame
            entries.append(struct.pack('<BBBBHHIH', *row[:6], len(frame), index + 1))
        resources[(14 << 16) | 1] = struct.pack('<HHH', 0, 1, count) + b''.join(entries)
        for corrupted in (False, True):
            with self.subTest(corrupted=corrupted):
                payloads = dict(resources)
                if corrupted:
                    payloads[(3 << 16) | 1] = b'wrong icon'
                buffers = {key: ctypes.create_string_buffer(value) for key, value in payloads.items()}
                kernel = Mock()
                kernel.LoadLibraryExW.return_value = 123
                kernel.FindResourceW.side_effect = lambda module, name, kind: (kind.value << 16) | name.value
                kernel.SizeofResource.side_effect = lambda module, resource: len(payloads[resource])
                kernel.LoadResource.side_effect = lambda module, resource: resource
                kernel.LockResource.side_effect = lambda resource: ctypes.addressof(buffers[resource])
                with patch.object(ctypes, 'WinDLL', create=True, return_value=kernel):
                    if corrupted:
                        with self.assertRaisesRegex(AssertionError, 'different logo'):
                            module.verify_windows_icon(Path('Scout-Studio.exe'), icon)
                    else:
                        module.verify_windows_icon(Path('Scout-Studio.exe'), icon)
                self.assertEqual(kernel.LoadLibraryExW.call_args.args[-1], 0x22)
                kernel.FreeLibrary.assert_called_once_with(123)

    def test_failed_app_prints_report_and_preserves_exit_failure(self):
        module = checker()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / 'smoke-report.json'
            def fail(*args, **kwargs):
                report.write_text(json.dumps(dict(ok=False, error='torchvision::nms missing')))
                kwargs['stdout'].write('native extension load failed\n')
                raise subprocess.CalledProcessError(1, args)
            with patch.object(module, 'run', side_effect=fail), contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(subprocess.CalledProcessError):
                    module.exercise_installed_app(root/'Scout-Studio', report, '1.0.0', root, {})
            self.assertIn('torchvision::nms missing', output.getvalue())
            self.assertIn('native extension load failed', output.getvalue())
            self.assertTrue((root / 'smoke-process.log').is_file())

    def test_success_without_new_report_cannot_reuse_stale_success(self):
        module = checker()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / 'smoke-report.json'
            report.write_text(json.dumps(dict(ok=True, frozen=True, version='1.0.0')))
            with patch.object(module, 'run'), self.assertRaises(FileNotFoundError):
                module.exercise_installed_app(root/'Scout-Studio', report, '1.0.0', root, {})

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
