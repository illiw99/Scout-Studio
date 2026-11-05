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
        checks = dict.fromkeys(('dependencies', 'ffmpeg', 'export', 'gui', 'playback', 'cli', 'worker', 'inference'), True)
        report = dict(ok=True, version='1.0.0', frozen=True, checks=checks)
        validate(report, '1.0.0')
        for replacement in ({'ok': False}, {'frozen': False}, {'version': '0.9.0'},
                            {'checks': {'gui': True}}):
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                validate({**report, **replacement}, '1.0.0')


if __name__ == '__main__':
    unittest.main()

class UpgradeValidationTests(unittest.TestCase):
    def fixture(self, root, version, system='Linux'):
        path = root / version
        path.mkdir()
        name = f'Scout-Studio-{version}-linux-x86_64.deb'
        installer = path / name
        installer.write_bytes(('fixture ' + version).encode())
        import hashlib
        (path / (name + '.sha256')).write_text(hashlib.sha256(installer.read_bytes()).hexdigest())
        (path / 'build-info.json').write_text(json.dumps(dict(version=version, platform=system,
             architecture='x86_64', installer=name)))
        return path

    def test_old_new_install_order_and_integrity_checks(self):
        module = checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = self.fixture(root, '1.0.0'), self.fixture(root, '1.1.0')
            protected = root / 'user-data'
            protected.mkdir()
            (protected / 'review.mlt').write_text('user edit')
            calls = []
            install = lambda path, replacing: calls.append(('install', path.parent.name, replacing))
            smoke = lambda version, legacy: calls.append(('smoke', version, legacy))
            result = module.upgrade(old, new, install, smoke, protected)
            self.assertTrue(result['upgrade_passed'])
            self.assertFalse(result['uninstall_preserved_data'])
            self.assertEqual(calls, [('install', '1.0.0', False), ('smoke', '1.0.0', True),
                                    ('install', '1.1.0', True), ('smoke', '1.1.0', False)])
            def corrupt(path, replacing):
                if replacing:
                    (protected / 'review.mlt').write_text('lost edit')
            with self.assertRaisesRegex(ValueError, 'protected'):
                module.upgrade(old, new, corrupt, smoke, protected)

    def test_bad_baseline_rejected_before_any_install(self):
        module = checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = self.fixture(root, '1.2.0'), self.fixture(root, '1.1.0')
            install = Mock()
            with self.assertRaisesRegex(ValueError, 'newer'):
                module.upgrade(old, new, install, Mock(), root)
            install.assert_not_called()
            info = json.loads((old / 'build-info.json').read_text())
            info['platform'] = 'Windows'
            (old / 'build-info.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'platform'):
                module.upgrade(old, new, install, Mock(), root)
            install.assert_not_called()

class WindowsPreflightTests(unittest.TestCase):
    def test_fixed_app_id_registration_or_shortcuts_prevent_installs(self):
        import sys
        import types
        module = checker()
        registry = types.SimpleNamespace(HKEY_CURRENT_USER=1, HKEY_LOCAL_MACHINE=2,
            KEY_WOW64_32KEY=32, KEY_WOW64_64KEY=64, KEY_READ=1,
            OpenKey=Mock(side_effect=FileNotFoundError()))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, winreg=registry), \
             patch.dict(module.os.environ, {'LOCALAPPDATA': directory}, clear=True):
            module.windows_preflight()
            self.assertEqual(registry.OpenKey.call_count, 4)
            registry.OpenKey.side_effect = None
            registry.OpenKey.return_value.__enter__ = Mock()
            registry.OpenKey.return_value.__exit__ = Mock(return_value=False)
            with self.assertRaisesRegex(RuntimeError, 'registration'):
                module.windows_preflight()
            registry.OpenKey.side_effect = FileNotFoundError()
            existing = Path(directory) / 'Programs/Scout-Studio'
            existing.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'files or shortcuts'):
                module.windows_preflight()

class NativePreservationTests(unittest.TestCase):
    def test_native_data_settings_and_pack_pointer_are_checked(self):
        module = checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / 'game'
            protected.mkdir()
            (protected / 'review.mlt').write_text('user correction')
            locations = module.application_locations('Linux', home=root, environ={})
            self.assertEqual(locations[0], root / '.local/share/Scout/Scout Studio')
            self.assertEqual(Path(locations[1]), root / '.config/Scout/Studio.conf')
            with patch.object(module, 'application_locations', return_value=locations):
                data, settings = module.seed_application_data('Linux', protected)
                before = module.preservation_snapshot(protected, data, settings, 'Linux')
                self.assertIn('app-data/model-packs/active.json', before)
                (data / 'model-packs/active.json').unlink()
                self.assertNotEqual(module.preservation_snapshot(protected, data, settings, 'Linux'), before)
                with self.assertRaisesRegex(RuntimeError, 'existing'):
                    module.seed_application_data('Linux', protected)
            windows = module.application_locations('Windows', home=root, environ={'LOCALAPPDATA': str(root / 'Local')})
            self.assertEqual(windows[0], root / 'Local/Scout/Scout Studio')
            self.assertEqual(windows[1], r'HKEY_CURRENT_USER\Software\Scout\Studio')
            mac = module.application_locations('Darwin', home=root, environ={})
            self.assertEqual(Path(mac[1]), root / 'Library/Preferences/com.scout.Studio.plist')
