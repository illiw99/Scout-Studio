"""The shipped Windows icon must be our SVG logo at standard shell sizes."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import Mock, patch

from scout.desktop import identity


class ApplicationIdentityTests(unittest.TestCase):
    def test_windows_taskbar_id_matches_installer_shortcuts(self):
        function = Mock(return_value=0)
        shell = Mock(SetCurrentProcessExplicitAppUserModelID=function)
        with patch.object(identity.sys, 'platform', 'win32'), \
             patch.object(identity.ctypes, 'WinDLL', create=True, return_value=shell):
            self.assertTrue(identity.configure_windows_app_id())
        function.assert_called_once_with(identity.WINDOWS_APP_ID)
        installer = Path(__file__).resolve().parents[1] / 'packaging/windows.iss'
        self.assertEqual(installer.read_text().count(f'AppUserModelID: "{identity.WINDOWS_APP_ID}"'), 2)

    def test_failed_windows_identity_can_be_reported_without_startup_exception(self):
        shell = Mock(SetCurrentProcessExplicitAppUserModelID=Mock(return_value=-1))
        with patch.object(identity.sys, 'platform', 'win32'), \
             patch.object(identity.ctypes, 'WinDLL', create=True, return_value=shell):
            self.assertFalse(identity.configure_windows_app_id())

    def test_other_platforms_do_not_load_windows_libraries(self):
        with patch.object(identity.sys, 'platform', 'linux'), \
             patch.object(identity.ctypes, 'WinDLL', create=True) as library:
            self.assertTrue(identity.configure_windows_app_id())
            self.assertEqual(identity.application_icon_path().name, 'scout.svg')
            library.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'desktop extra not installed')
class ApplicationIconTests(unittest.TestCase):
    def test_icon_sizes_decode_and_match_the_vector_source(self):
        from PySide6.QtGui import QImage
        assets = Path(identity.__file__).parent
        data = (assets / 'scout.ico').read_bytes()
        self.assertEqual(struct.unpack_from('<HHH', data), (0, 1, 7))
        source_hash = hashlib.sha256((assets / 'scout.svg').read_bytes()).hexdigest()
        sizes = []
        for index in range(7):
            width, height, _, _, planes, bits, length, offset = struct.unpack_from('<BBBBHHII', data, 6 + 16 * index)
            width, height = width or 256, height or 256
            image = QImage.fromData(data[offset:offset + length], 'PNG')
            self.assertFalse(image.isNull())
            self.assertEqual((image.width(), image.height()), (width, height))
            self.assertEqual((planes, bits), (1, 32))
            self.assertEqual(image.text('source-svg-sha256'), source_hash)
            self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
            sizes.append(width)
        self.assertEqual(sizes, [16, 24, 32, 48, 64, 128, 256])


if __name__ == '__main__':
    unittest.main()
