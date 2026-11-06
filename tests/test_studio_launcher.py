"""A locally selected standalone runtime preserves workspace and quoted arguments."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('bash'), 'bash launcher check')
class StudioLauncherTests(unittest.TestCase):
    def test_selected_runtime_with_spaces_receives_workspace_and_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source workspace';root.mkdir()
            launcher=root/'start-studio.sh';shutil.copyfile('start-studio.sh',launcher)
            executable=root/'standalone app';executable.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');executable.chmod(0o700)
            (root/'.scout-studio-runtime').write_text(str(executable)+'\n')
            result=subprocess.run(['bash',str(launcher),'--test','value with spaces'],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout),['--workspace',str(root),'--test','value with spaces'])
            executable.unlink()
            failed=subprocess.run(['bash',str(launcher)],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('configured Scout Studio runtime is unavailable',failed.stderr)
