"""Disk image retries never publish partial or unverified installers."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
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
        resource = self.stage / 'Scout Studio.app/Contents/Resources/fixture.txt'
        resource.parent.mkdir(parents=True)
        resource.write_text('temporary installer fixture')
        (self.stage / 'Applications').symlink_to('/Applications')
        self.module.create_dmg(self.stage, self.installer, self.root / 'logs')
        self.assertGreater(self.installer.stat().st_size, 0)


class PackagePathTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('dpkg-deb'), 'Debian package tools unavailable')
    def test_linux_package_contains_canonical_launch_paths(self):
        module = builder()
        with tempfile.TemporaryDirectory(prefix='scout-studio-package-') as folder:
            root = Path(folder)
            (root / 'scout/desktop').mkdir(parents=True)
            (root / 'scout/desktop/scout.svg').write_text('<svg/>')
            (root / 'scripts').mkdir()
            (root / 'scripts/test-installer.py').write_text('# fixture')
            bundle = root / 'build/installer/dist/Scout-Studio'
            bundle.mkdir(parents=True)
            (bundle / 'Scout-Studio').write_text('#!/bin/sh\nexit 0\n')
            (bundle / 'Scout-Studio').chmod(0o755)
            actual_run = module.run
            commands = []
            def run(*args):
                commands.append(args)
                if args[0] == 'dpkg-deb':
                    actual_run(*args)
            with patch.object(module, 'ROOT', root), \
                 patch.object(module, 'project_version', return_value='1.2.3'), \
                 patch.object(module.platform, 'system', return_value='Linux'), \
                 patch.object(module.platform, 'machine', return_value='x86_64'), \
                 patch.object(module, 'run', side_effect=run), \
                 patch.dict(os.environ), patch.object(sys, 'argv', ['build-installers.py']), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.build()
                self.assertEqual(os.environ['SCOUT_STUDIO_BUILD_VERSION'], '1.2.3')
            self.assertEqual(Path(commands[0][-1]).name, 'scout-studio.spec')
            installer = root / 'dist/installers/Scout-Studio-1.2.3-linux-x86_64.deb'
            self.assertTrue(installer.is_file())
            package = subprocess.check_output(['dpkg-deb', '-f', str(installer), 'Package'], text=True)
            self.assertEqual(package.strip(), 'scout-studio')
            unpack = root / 'unpacked'
            subprocess.run(['dpkg-deb', '-x', str(installer), str(unpack)], check=True)
            self.assertEqual(os.readlink(unpack / 'usr/bin/Scout-Studio'),
                             '/opt/Scout-Studio/Scout-Studio')
            self.assertTrue((unpack / 'opt/Scout-Studio/Scout-Studio').is_file())
            desktop = (unpack / 'usr/share/applications/scout-studio.desktop').read_text()
            self.assertIn('Name=Scout Studio\n', desktop)
            self.assertIn('Exec=/opt/Scout-Studio/Scout-Studio\n', desktop)
            self.assertIn('Icon=scout-studio\n', desktop)
            self.assertTrue((unpack / 'usr/share/icons/hicolor/scalable/apps/scout-studio.svg').is_file())

    def test_frozen_names_and_windows_upgrade_identity(self):
        root = Path(__file__).resolve().parents[1]
        spec = ast.parse((root / 'packaging/scout-studio.spec').read_text())
        names = {}
        for node in ast.walk(spec):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                for keyword in node.keywords:
                    if keyword.arg in {'name', 'bundle_identifier'} and isinstance(keyword.value, ast.Constant):
                        names.setdefault(node.func.id, []).append(keyword.value.value)
        self.assertEqual(names['EXE'], ['Scout-Studio', 'Scout-Studio-CLI'])
        self.assertEqual(names['COLLECT'], ['Scout-Studio'])
        self.assertEqual(names['BUNDLE'], ['Scout Studio.app', 'org.scoutstudio.studio'])
        windows = (root / 'packaging/windows.iss').read_text()
        self.assertIn('AppId={{F82B9577-D75C-4EBB-97A6-CB51CF2E498C}', windows)
        self.assertIn('DefaultDirName={localappdata}\\Programs\\Scout-Studio', windows)
        self.assertIn('UsePreviousAppDir=yes', windows)
        self.assertIn('UsePreviousGroup=no', windows)
        self.assertIn('UninstallDisplayName=Scout Studio', windows)


if __name__ == '__main__':
    unittest.main()
