"""Pinned baseline fetches validate metadata before accepting installer bytes."""
import importlib.util
import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


def helper():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare-upgrade.py'
    spec = importlib.util.spec_from_file_location('prepare_upgrade', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareUpgradeTests(unittest.TestCase):
    def test_pinned_matching_release_is_recorded_and_wrong_channel_rejected(self):
        module = helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'build-info.json').write_text(json.dumps(dict(version='1.1.0', platform='Linux',
                architecture='x86_64', installer='Scout-Studio-1.1.0-linux-x86_64.deb')))
            name = 'Scout-Studio-1.0.0-linux-x86_64.deb'
            release = dict(tag_name='v1.0.0', draft=False, prerelease=False, assets=[dict(name=name,
                size=7, browser_download_url=f'https://github.com/{module.REPOSITORY}/releases/download/v1.0.0/{name}')])
            def download(update, directory):
                self.assertEqual(update.sha256, 'a' * 64)
                path = directory / update.filename
                path.write_bytes(b'fixture')
                return path
            read = lambda *_: json.dumps(release).encode()
            module.prepare(root, '1.0.0', 'linux-x86_64', 'a' * 64, read=read, download=download)
            info = json.loads((root / 'build-info.json').read_text())
            self.assertEqual(info['upgrade_baseline']['installer'], name)
            self.assertEqual(info['upgrade_baseline']['size'], 7)
            release['draft'] = True
            no_download = Mock()
            with self.assertRaisesRegex(ValueError, 'stable'):
                module.prepare(root, '1.0.0', 'linux-x86_64', 'a' * 64, read=read, download=no_download)
            no_download.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'pinned'):
                module.prepare(root, '1.0.0', 'linux-x86_64', '', read=read, download=no_download)

    def test_ci_token_is_confined_to_fixed_api_without_redirects(self):
        module=helper();opener=Mock();opener.open.return_value=io.BytesIO(b'{}')
        url=f'https://api.github.com/repos/{module.REPOSITORY}/releases/tags/v1.0.2'
        with patch.dict(os.environ,GH_TOKEN='fixture-token'),patch.object(module,'build_opener',return_value=opener):
            self.assertEqual(module.release_read(url,100),b'{}')
            request=opener.open.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'),'Bearer fixture-token')
            with self.assertRaises(ValueError):module.release_read('https://evil.test/releases/tags/v1.0.2',100)
            opener.open.assert_called_once()
        self.assertIsNone(module._NoTokenRedirect().redirect_request(None,None,None,None,None,None))
