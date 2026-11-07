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
from types import SimpleNamespace
from unittest.mock import Mock, patch


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

class SigningTests(unittest.TestCase):
    def test_required_signing_fails_closed_and_partial_mac_configuration_rejected(self):
        module = builder()
        with patch.dict(os.environ, {'SCOUT_REQUIRE_SIGNING': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'required'):
                module.signing_configuration('Windows')
            self.assertFalse(module.signing_configuration('Linux'))
        with patch.dict(os.environ, {'SCOUT_MAC_SIGN_IDENTITY': 'Developer ID test'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'both'):
                module.signing_configuration('Darwin')
        with patch.dict(os.environ, {'SCOUT_WINDOWS_CERT_SHA1': 'A' * 40}, clear=True):
            self.assertTrue(module.signing_configuration('Windows'))

    def test_windows_signature_verified_after_signing_and_failure_propagates(self):
        module = builder()
        with patch.dict(os.environ, {'SCOUT_WINDOWS_CERT_SHA1': 'A' * 40}, clear=True), \
             patch.object(module, 'run') as run:
            module.sign_windows(Path('fixture.exe'))
            self.assertEqual(run.call_args_list[0].args[:3], ('signtool', 'sign', '/fd'))
            self.assertEqual(run.call_args_list[1].args, ('signtool', 'verify', '/pa', '/all', Path('fixture.exe')))
            run.side_effect = subprocess.CalledProcessError(1, ['signtool'])
            with self.assertRaises(subprocess.CalledProcessError):
                module.sign_windows(Path('fixture.exe'))


class BaselineBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = builder()

    def test_paired_arguments_required_before_any_build_work(self):
        for arguments in (['--baseline-pack', 'pack.zip'], ['--baseline-sha256', 'a' * 64]):
            with self.subTest(arguments=arguments), \
                 patch.object(sys, 'argv', ['build-installers.py', *arguments]), \
                 patch.object(self.module, 'run') as run, \
                 patch.object(self.module, 'project_version') as version, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.module.build()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
                version.assert_not_called()

    def test_version_only_does_not_prepare_or_build(self):
        helper = Mock()
        with patch.object(sys, 'argv', ['build-installers.py', '--version-only']), \
             patch.object(self.module, 'project_version', return_value='1.2.3'), \
             patch.object(self.module, 'run') as run, \
             patch.dict(sys.modules, {'scout.model_defaults': SimpleNamespace(prepare_bundle=helper)}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.module.build()
        self.assertEqual(output.getvalue(), '1.2.3\n')
        run.assert_not_called()
        helper.assert_not_called()

    def build_isolated(self, *, baseline=True, failure=False, ffmpeg=False, ffmpeg_invalid=False):
        source = self.root / 'source'
        (source / 'scripts').mkdir(parents=True)
        (source / 'scripts/test-installer.py').write_text('# fixture')
        work, output = self.root / 'isolated/work', self.root / 'isolated/output'
        archive = self.root / 'private.zip'
        archive.write_bytes(b'fixture archive')
        arguments = ['--work-dir', str(work), '--output-dir', str(output),
                     '--runtime-profile', 'inference']
        if baseline:
            arguments += ['--baseline-pack', str(archive), '--baseline-sha256', 'a' * 64]
        ffmpeg_source = self.root / 'custom-ffmpeg-build'
        if ffmpeg:
            if not ffmpeg_invalid:
                ffmpeg_source.write_bytes(b'fixture executable')
                ffmpeg_source.chmod(0o755)
            arguments += ['--ffmpeg-binary', str(ffmpeg_source)]
        staged_media = []
        staged = []
        def prepare(pack, digest, destination):
            self.assertEqual(pack, archive.resolve())
            self.assertEqual(digest, 'a' * 64)
            self.assertFalse(destination.exists())
            if failure:
                raise ValueError('invalid baseline fixture')
            destination.mkdir()
            (destination / 'model-pack.zip').write_bytes(pack.read_bytes())
            (destination / 'receipt.json').write_text('{}')
            staged.append(destination)
            return destination
        helper = Mock(side_effect=prepare)
        commands = []
        def run(*args):
            commands.append(args)
            if args[1:3] == ('-m', 'PyInstaller'):
                self.assertEqual(args[args.index('--distpath') + 1], work.resolve() / 'dist')
                self.assertEqual(args[args.index('--workpath') + 1], work.resolve() / 'pyinstaller')
                if ffmpeg:
                    media = Path(os.environ['SCOUT_FFMPEG_BINARY'])
                    staged_media.append(media)
                    self.assertEqual(media.name, 'ffmpeg.exe')
                    self.assertEqual(media.read_bytes(), ffmpeg_source.read_bytes())
                    self.assertNotEqual(media, ffmpeg_source)
                    self.assertTrue(os.access(media, os.X_OK))
                else:
                    self.assertNotIn('SCOUT_FFMPEG_BINARY', os.environ)
                if baseline:
                    self.assertEqual(Path(os.environ['SCOUT_BASELINE_DIR']), staged[0])
                    self.assertTrue((staged[0] / 'receipt.json').is_file())
                else:
                    self.assertNotIn('SCOUT_BASELINE_DIR', os.environ)
            else:
                (output / 'Scout-Studio-1.2.3-windows-x86_64.exe').write_bytes(b'installer fixture')
        with patch.object(self.module, 'ROOT', source), \
             patch.object(self.module, 'project_version', return_value='1.2.3'), \
             patch.object(self.module.platform, 'system', return_value='Windows'), \
             patch.object(self.module.platform, 'machine', return_value='amd64'), \
             patch.object(self.module, 'run', side_effect=run), \
             patch.object(self.module.shutil, 'which', return_value='iscc'), \
             patch.dict(sys.modules, {'scout.model_defaults': SimpleNamespace(prepare_bundle=helper)}), \
             patch.dict(os.environ, {'SCOUT_BASELINE_DIR': 'ambient-stale-input',
                                     'SCOUT_FFMPEG_BINARY': 'ambient-ffmpeg'}, clear=True), \
             patch.object(sys, 'argv', ['build-installers.py', *arguments]), \
             contextlib.redirect_stdout(io.StringIO()):
            if failure or ffmpeg_invalid:
                with self.assertRaisesRegex(ValueError, 'regular executable' if ffmpeg_invalid else 'invalid baseline'):
                    self.module.build()
                self.assertEqual(commands, [])
            else:
                self.module.build()
                self.assertEqual(len(commands), 2)
                self.assertTrue((output / 'build-info.json').is_file())
            self.assertEqual(os.environ['SCOUT_BASELINE_DIR'], 'ambient-stale-input')
            self.assertEqual(os.environ['SCOUT_FFMPEG_BINARY'], 'ambient-ffmpeg')
        self.assertFalse((source / 'build').exists())
        self.assertFalse((source / 'dist').exists())
        self.assertEqual(list(work.glob('baseline-*')), [])
        self.assertEqual(list(work.glob('ffmpeg-*')), [])
        self.assertTrue(all(not path.exists() for path in staged_media))
        self.assertEqual(helper.call_count, 1 if baseline and not ffmpeg_invalid else 0)

    def test_baseline_is_prepared_before_freezing_with_isolated_paths(self):
        self.build_isolated()

    @unittest.skipIf(sys.platform == 'win32', 'symlink creation requires Windows privileges')
    def test_build_paths_through_symlink_are_resolved(self):
        # macOS temporary paths use /var, a symlink to /private/var.
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        self.root = alias
        self.build_isolated(ffmpeg=True)

    def test_explicit_ffmpeg_is_staged_with_normalized_name(self):
        self.build_isolated(ffmpeg=True)

    def test_missing_ffmpeg_prevents_build(self):
        self.build_isolated(ffmpeg=True, ffmpeg_invalid=True)

    def test_baseline_validation_failure_prevents_build(self):
        self.build_isolated(failure=True)

    def test_no_baseline_does_not_bundle_ambient_input(self):
        self.build_isolated(baseline=False)

    def test_spec_includes_exact_baseline_files_and_rejects_missing_files(self):
        source = Path(__file__).resolve().parents[1] / 'packaging/scout-studio.spec'
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                  target.id == 'baseline_dir' for target in node.targets)) or
                 (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and
                  node.test.id == 'baseline_dir')]
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec')
        for name in ('model-pack.zip', 'receipt.json', 'private-unrelated.pt'):
            (self.root / name).write_bytes(b'fixture')
        with patch.dict(os.environ, {'SCOUT_BASELINE_DIR': str(self.root)}):
            namespace = dict(os=os, Path=Path, datas=[])
            exec(code, namespace)
            self.assertEqual(namespace['datas'], [
                (str(self.root / 'model-pack.zip'), '_scout_baseline'),
                (str(self.root / 'receipt.json'), '_scout_baseline')])
            (self.root / 'receipt.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing verified baseline'):
                exec(code, dict(os=os, Path=Path, datas=[]))
        with patch.dict(os.environ, {}, clear=True):
            namespace = dict(os=os, Path=Path, datas=[])
            exec(code, namespace)
            self.assertEqual(namespace['datas'], [])


    def test_spec_ffmpeg_and_optional_linux_cursor_shim(self):
        source = Path(__file__).resolve().parents[1] / 'packaging/scout-studio.spec'
        tree = ast.parse(source.read_text())
        start = next(index for index, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                        target.id == 'ffmpeg_binary' for target in node.targets))
        code = compile(ast.Module(body=tree.body[start:start + 3], type_ignores=[]), str(source), 'exec')
        ffmpeg = self.root / 'ffmpeg'
        ffmpeg.write_bytes(b'fixture executable')
        ffmpeg.chmod(0o755)
        prefix = self.root / 'private-prefix'
        shim = prefix / 'lib/scout-native/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0'
        shim.parent.mkdir(parents=True)
        shim.write_bytes(b'fixture library')
        with patch.dict(os.environ, {'SCOUT_FFMPEG_BINARY': str(ffmpeg)}, clear=True):
            namespace = dict(os=os, sys=SimpleNamespace(platform='linux', prefix=str(prefix)), Path=Path, binaries=[])
            exec(code, namespace)
            self.assertEqual(namespace['binaries'], [(str(ffmpeg), '_scout_media'),
                (str(shim), 'lib/scout-native/usr/lib/x86_64-linux-gnu')])
            ffmpeg.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'non-executable'):
                exec(code, namespace)
            ffmpeg.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                exec(code, namespace)
        with patch.dict(os.environ, {}, clear=True):
            namespace = dict(os=os, sys=SimpleNamespace(platform='win32', prefix=str(prefix)), Path=Path, binaries=[])
            exec(code, namespace)
            self.assertEqual(namespace['binaries'], [])
            shim.unlink()
            namespace['sys'].platform = 'linux'
            exec(code, namespace)
            self.assertEqual(namespace['binaries'], [])
