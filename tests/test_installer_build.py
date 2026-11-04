"""Disk image retries never publish partial or unverified installers."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/build-installers.py'
    spec = importlib.util.spec_from_file_location('installer_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiskImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        self.installer = self.root / 'release.dmg'
        self.installer.write_bytes(b'previous verified image')
        self.module = builder()

    def test_busy_retry_discards_partial_then_verifies_before_publishing(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if command[1] == 'create':
                target = Path(command[-1])
                if len(commands) == 1:
                    target.write_bytes(b'partial')
                    kwargs['stdout'].write('hdiutil: create failed - Resource busy\n')
                    return subprocess.CompletedProcess(command, 1)
                self.assertFalse(Path(commands[0][-1]).exists())
                target.write_bytes(b'complete image')
            else:
                self.assertEqual(command[1], 'verify')
                self.assertEqual(self.installer.read_bytes(), b'previous verified image')
            return subprocess.CompletedProcess(command, 0)
        with patch.object(self.module.subprocess, 'run', side_effect=run), \
             patch.object(self.module.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        self.assertEqual(self.installer.read_bytes(), b'complete image')
        self.assertEqual([command[1] for command in commands], ['create', 'create', 'verify'])
        sleep.assert_called_once_with(2)
        self.assertEqual(list(self.root.glob('.dmg-build-*')), [])

    def test_non_busy_error_is_not_retried_and_preserves_existing_installer(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('hdiutil: create failed - No space left on device\n')
            return subprocess.CompletedProcess(command, 1)
        with patch.object(self.module.subprocess, 'run', side_effect=fail) as run, \
             patch.object(self.module.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        self.assertIn('No space left', error.exception.output)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.installer.read_bytes(), b'previous verified image')

    def test_persistent_busy_error_stops_after_five_attempts(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('hdiutil: create failed - Resource busy\n')
            return subprocess.CompletedProcess(command, 1)
        with patch.object(self.module.subprocess, 'run', side_effect=fail) as run, \
             patch.object(self.module.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        self.assertEqual(run.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8, 16])
        self.assertEqual(self.installer.read_bytes(), b'previous verified image')

    def test_verification_failure_cannot_publish_the_created_image(self):
        def fail(command, **kwargs):
            if command[1] == 'create':
                Path(command[-1]).write_bytes(b'corrupt image')
                return subprocess.CompletedProcess(command, 0)
            raise subprocess.CalledProcessError(1, command)
        with patch.object(self.module.subprocess, 'run', side_effect=fail), \
             patch.object(self.module.time, 'sleep') as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        sleep.assert_not_called()
        self.assertEqual(self.installer.read_bytes(), b'previous verified image')

    @unittest.skipUnless(sys.platform == 'darwin', 'native macOS disk-image check')
    def test_native_macos_image_creation_and_verification(self):
        resource = self.stage / 'PlayChop.app/Contents/Resources/fixture.txt'
        resource.parent.mkdir(parents=True)
        resource.write_text('temporary installer fixture')
        (self.stage / 'Applications').symlink_to('/Applications')
        self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        self.assertGreater(self.installer.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
