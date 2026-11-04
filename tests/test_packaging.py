"""Packaging contracts that do not require Qt or a build toolchain."""

from __future__ import annotations

import contextlib
import ast
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scout.desktop import main


class DesktopEntryPointTests(unittest.TestCase):
    def test_missing_qt_explains_desktop_extra(self):
        error = io.StringIO()
        with patch.dict(sys.modules, {'scout.desktop.app': None, 'PySide6': None}):
            # Block Qt at the import boundary without masking other dependencies.
            import builtins
            original = builtins.__import__

            def missing_qt(name, *args, **kwargs):
                if name == 'app':
                    raise ModuleNotFoundError("No module named 'PySide6'", name='PySide6')
                return original(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=missing_qt), contextlib.redirect_stderr(error):
                self.assertEqual(main([]), 1)
        self.assertIn('python -m pip install "scout-flag[desktop]"', error.getvalue())

    def test_unrelated_missing_dependency_is_not_hidden(self):
        import builtins
        original = builtins.__import__

        def missing_dependency(name, *args, **kwargs):
            if name == 'app':
                raise ModuleNotFoundError("No module named 'broken_dependency'", name='broken_dependency')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=missing_dependency):
            with self.assertRaisesRegex(ModuleNotFoundError, 'broken_dependency'):
                main([])


class ArtifactPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/check-package.py'
        spec = importlib.util.spec_from_file_location('check_package', path)
        cls.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checker)

    def test_rejects_user_data_and_unsafe_paths(self):
        for name in ('models/play.pkl', 'media/game.mp4', 'fields/game.json',
                     '.env', '../secret', '/secret', 'scout/__pycache__/app.pyc',
                     'Scout/preferences.ini', '.scout/preferences.ini',
                     'scout/config.json', 'scout_flag.egg-info/weights.pt'):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    self.checker.validate_members([name], wheel=False)

    def test_accepts_required_assets(self):
        self.checker.validate_members(['scout/py.typed', 'scout/desktop/scout.svg',
                                       'scout/__init__.py'], wheel=True)


@unittest.skipUnless(importlib.util.find_spec('PyInstaller'), 'installer build tools are not installed')
class NativeVisionBundleTests(unittest.TestCase):
    def test_installed_torchvision_operator_libraries_enter_binary_collection(self):
        from PyInstaller.utils.hooks import collect_dynamic_libs
        spec = Path(__file__).resolve().parents[1] / 'packaging/playchop.spec'
        tree = ast.parse(spec.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'binaries'
                                  for target in node.targets))
        binaries = eval(compile(ast.Expression(assignment.value), str(spec), 'eval'),
                        {'collect_dynamic_libs': collect_dynamic_libs})
        operators = [(Path(source), destination) for source, destination in binaries
                     if Path(source).name.startswith('_C')]
        self.assertTrue(operators, 'TorchVision native NMS operators are absent from the bundle')
        for source, destination in operators:
            self.assertTrue(source.is_file())
            self.assertEqual(destination, 'torchvision')


if __name__ == '__main__':
    unittest.main()
