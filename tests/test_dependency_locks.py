"""Release snapshots must be complete hash-pinned target resolutions."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dependency_locks", ROOT / "scripts/dependency-locks.py")
locks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(locks)


class DependencyLocksTests(unittest.TestCase):
    def test_all_target_snapshots(self):
        for target in locks.TARGETS:
            with self.subTest(target=target):
                locks.validate(target, (locks.LOCKS / f"{target}.txt").read_text())

    def test_native_cpu_and_cuda_snapshots_are_separate_and_hash_pinned(self):
        for system in ('linux', 'windows'):
            cpu = f'{system}-x86_64-cpu'
            cuda = f'{system}-x86_64-cu128'
            cpu_text = (locks.LOCKS / (cpu + '.txt')).read_text()
            cuda_text = (locks.LOCKS / (cuda + '.txt')).read_text()
            self.assertEqual(locks.pins(cpu_text)['torch'], '2.7.1+cpu')
            self.assertEqual(locks.pins(cuda_text)['torch'], '2.7.1+cu128')
            self.assertNotEqual(cpu_text, cuda_text)
            with self.assertRaisesRegex(ValueError, 'CUDA dependency'):
                locks.validate(cpu, cpu_text + '\nnvidia-cuda-runtime-cu12==12.8.90 --hash=sha256:' + 'a' * 64 + '\n')

    def test_requirements_reject_unpinned_unhashed_and_duplicate_entries(self):
        for text in ("numpy>=1.24", "numpy==1.26.4", "numpy==1.26.4 --hash=sha256:bad",
                     "numpy==1.26.4 --hash=sha256:" + "a" * 64 + "\nnumpy==1.26.4 --hash=sha256:" + "b" * 64):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                locks.pins(text)

    def test_intel_numpy_and_windows_cuda_contracts(self):
        for target, old, new in (("macos-x86_64", "numpy==1.26.4", "numpy==2.5.4"),
                                 ("windows-x86_64-cu128", "torch==2.7.1+cu128", "torch==2.7.1")):
            text = (locks.LOCKS / f"{target}.txt").read_text()
            self.assertIn(old, text)
            with self.assertRaises(ValueError):
                locks.validate(target, text.replace(old, new))

    def test_target_marker_dependencies(self):
        for target, name in (("windows-x86_64-cu128", "pefile"), ("macos-arm64", "macholib")):
            text = (locks.LOCKS / f"{target}.txt").read_text()
            text = text.replace(name + "==", "wrong-package==")
            with self.assertRaises(ValueError):
                locks.validate(target, text)

    def test_input_changes_require_regeneration(self):
        target = "linux-x86_64-cpu"
        text = (locks.LOCKS / f"{target}.txt").read_text()
        with self.assertRaises(ValueError):
            locks.validate(target, text.replace("input-sha256=", "outdated-input="))

    def test_extra_indexes_are_rejected(self):
        target = "macos-arm64"
        text = (locks.LOCKS / f"{target}.txt").read_text()
        with self.assertRaises(ValueError):
            locks.validate(target, text + "\n--extra-index-url https://example.invalid/simple\n")

    @unittest.skipUnless(importlib.util.find_spec('pip'), 'pip parser integration requires pip')
    def test_pip_retains_primary_and_torch_indexes_when_parsing_locks(self):
        from pip._internal.commands import create_command
        # Parsing requirements is read-only and does not resolve/download wheels.
        for target, backend in (('linux-x86_64-cpu', 'cpu'), ('windows-x86_64-cu128', 'cu128')):
            with self.subTest(target=target), patch.dict(locks.os.environ, {}, clear=True):
                command = create_command('install')
                options, arguments = command.parse_args(['-r', str(locks.LOCKS / (target + '.txt'))])
                with command._build_session(options) as session:
                    finder = command._build_package_finder(options=options, session=session)
                    command.get_requirements(arguments, options, finder, session)
                    self.assertEqual(finder.search_scope.index_urls,
                        ['https://pypi.org/simple', 'https://download.pytorch.org/whl/' + backend])
            text = (locks.LOCKS / (target + '.txt')).read_text()
            primary, extra = '--index-url https://pypi.org/simple', '--extra-index-url https://download.pytorch.org/whl/' + backend
            with self.assertRaisesRegex(ValueError, 'index options'):
                locks.validate(target, text.replace(primary + '\n' + extra, extra + '\n' + primary))

    def test_installer_refuses_global_or_wrong_platform_environment(self):
        with patch.object(locks.sys, 'prefix', locks.sys.base_prefix):
            with self.assertRaisesRegex(ValueError, 'new Python 3.12 virtual environment'):
                locks.install_locked('linux-x86_64-cpu')
        with patch.object(locks.sys, 'prefix', '/temporary/build-env'), \
             patch.object(locks.sys, 'version_info', (3, 12)), \
             patch.object(locks.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                locks.install_locked('linux-x86_64-cpu')

    def test_headless_replacement_keeps_exact_hash_requirements(self):
        captured = []
        def run(command, **kwargs):
            captured.append(command)
            self.assertIn('--isolated', command)
            self.assertEqual(kwargs['env']['PIP_CONFIG_FILE'], locks.os.devnull)
            if 'install' in command and '--no-deps' in command and '-r' in command:
                text = Path(command[-1]).read_text()
                self.assertIn('opencv-python-headless==', text)
                self.assertIn('--hash=sha256:', text)
                self.assertIn('--require-hashes', command)
                self.assertNotIn('opencv-python==', text)
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(locks.sys, 'prefix', directory), \
             patch.object(locks.sys, 'version_info', (3, 12)), \
             patch.object(locks.platform, 'system', return_value='Linux'), \
             patch.object(locks.platform, 'machine', return_value='x86_64'), \
             patch.object(locks.subprocess, 'run', side_effect=run), \
             patch.object(locks.metadata, 'distributions', return_value=[]), \
             patch.object(locks, 'check_installed') as check:
            (Path(directory)/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            locks.install_locked('linux-x86_64-cpu')
        check.assert_called_once_with('linux-x86_64-cpu')
        self.assertEqual(sum('--require-hashes' in call for call in captured), 2)

    def test_populated_and_shared_venvs_are_rejected_before_pip(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(locks.sys, 'prefix', directory), \
             patch.object(locks.sys, 'version_info', (3, 12)), \
             patch.object(locks.platform, 'system', return_value='Linux'), \
             patch.object(locks.platform, 'machine', return_value='x86_64'), \
             patch.object(locks.subprocess, 'run') as run:
            config=Path(directory)/'pyvenv.cfg'
            config.write_text('include-system-site-packages = false\n')
            with patch.object(locks.metadata, 'distributions',
                              return_value=[SimpleNamespace(metadata={'Name':'numpy'})]):
                with self.assertRaisesRegex(ValueError, 'populated'):
                    locks.install_locked('linux-x86_64-cpu')
            config.write_text('include-system-site-packages = true\n')
            with self.assertRaisesRegex(ValueError, 'isolated new'):
                locks.install_locked('linux-x86_64-cpu')
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
