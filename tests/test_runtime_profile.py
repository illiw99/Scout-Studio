import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scout import runtime


class RuntimeProfileTests(unittest.TestCase):
    def test_source_and_old_bundle_keep_reference_identity(self):
        with patch.object(runtime.sys, 'frozen', False, create=True):
            self.assertEqual(runtime.profile(), 'reference')
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime.sys, 'frozen', True, create=True), \
             patch.object(runtime.sys, '_MEIPASS', directory, create=True):
            self.assertEqual(runtime.profile(), 'reference')
            path = Path(directory) / 'runtime-profile.json'
            path.write_text(json.dumps(dict(schema=1, profile='inference')))
            self.assertEqual(runtime.profile(), 'inference')
            path.write_text(json.dumps(dict(schema=True, profile='inference')))
            with self.assertRaises(ValueError):
                runtime.profile()
