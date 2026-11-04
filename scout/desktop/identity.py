"""Shared native application identity and platform icon paths."""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys

WINDOWS_APP_ID = 'org.scoutstudio.studio'


def application_icon_path() -> Path:
    return Path(__file__).with_name('scout.ico' if sys.platform == 'win32' else 'scout.svg')


def configure_windows_app_id() -> bool:
    """Identify the GUI to Windows before Qt creates taskbar windows."""
    if sys.platform != 'win32':
        return True
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    function = shell.SetCurrentProcessExplicitAppUserModelID
    function.argtypes = [ctypes.c_wchar_p]
    function.restype = ctypes.c_long
    return function(WINDOWS_APP_ID) >= 0
