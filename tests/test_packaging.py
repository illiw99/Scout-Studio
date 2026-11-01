"""Packaging contracts that do not require Qt or a build toolchain."""

from __future__ import annotations

import contextlib
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
                     'scout/config.json', 'scout_flag.egg-info/weights.pt'):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    self.checker.validate_members([name], wheel=False)

    def test_accepts_required_assets(self):
        self.checker.validate_members(['scout/py.typed', 'scout/desktop/scout.svg',
                                       'scout/__init__.py'], wheel=True)


if __name__ == '__main__':
    unittest.main()
