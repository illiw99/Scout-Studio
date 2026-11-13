"""Update checks use fake HTTPS metadata and byte streams, never the network."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from scout.updates import check, stage, Update, _url


class UpdateTests(unittest.TestCase):
    def metadata(self):
        data = b'installer fixture'
        digest = hashlib.sha256(data).hexdigest()
        name = 'Scout-Studio-1.1.0-linux-x86_64.deb'
        base = 'https://github.com/illiw99/Scout-Studio/releases/download/v1.1.0/'
        context = dict(repository='illiw99/Scout-Studio', commit='a' * 40, run_id=1, run_attempt=1)
        manifest = dict(schema=1, tag='v1.1.0', context=context, platforms=[dict(
            candidate=dict(platform='linux-x86_64', version='1.1.0', installer=name, context=context,
                           files={name: dict(sha256=digest, size=len(data))}, signing='not-configured'),
            test_result=dict(context=context, platform='linux-x86_64', installer_sha256=digest,
                             install_and_uninstall_passed=True))])
        values = {'release-manifest.json': json.dumps(manifest).encode(),
                  'SHA256SUMS.txt': f'{digest}  {name}\n'.encode()}
        release = dict(tag_name='v1.1.0', draft=False, prerelease=False,
                       assets=[dict(name=key, size=len(value), browser_download_url=base + key)
                               for key, value in {**values, name: data}.items()])
        def read(url, limit):
            return json.dumps(release).encode() if url.endswith('/latest') else values[url.rsplit('/', 1)[-1]]
        return read, release, manifest, values, data

    def test_checks_identity_evidence_and_hash_before_staging(self):
        read, release, manifest, values, data = self.metadata()
        update = check('1.0.0', target='linux-x86_64', read=read)
        self.assertEqual(update.version, '1.1.0')
        self.assertIsNone(check('1.1.0', target='linux-x86_64', read=read))
        with tempfile.TemporaryDirectory() as directory:
            path = stage(update, directory, open_url=lambda _: io.BytesIO(data))
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(stage(update, directory), path)
        manifest['platforms'][0]['test_result']['install_and_uninstall_passed'] = False
        values['release-manifest.json'] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(ValueError, 'evidence'):
            check('1.0.0', target='linux-x86_64', read=read)

    def test_installer_only_release_binds_embedded_evidence_to_github_digest(self):
        read,release,manifest,values,data=self.metadata()
        release['assets']=[a for a in release['assets'] if a['name'].endswith('.deb')]
        release['assets'][0]['digest']='sha256:'+hashlib.sha256(data).hexdigest()
        release['body']='Human notes\n<!-- scout-installer-evidence-v1\n'+json.dumps(manifest)+'\n-->'
        self.assertEqual(check('1.0.0',target='linux-x86_64',read=read).sha256,hashlib.sha256(data).hexdigest())
        release['assets'][0]['digest']='sha256:'+'0'*64
        with self.assertRaisesRegex(ValueError,'digest'):check('1.0.0',target='linux-x86_64',read=read)
        release['assets'][0]['digest']='sha256:'+hashlib.sha256(data).hexdigest()
        manifest['platforms'][0]['test_result']['install_and_uninstall_passed']=False
        release['body']='<!-- scout-installer-evidence-v1\n'+json.dumps(manifest)+'\n-->'
        with self.assertRaisesRegex(ValueError,'evidence'):check('1.0.0',target='linux-x86_64',read=read)
        for body in ('', '<!-- scout-installer-evidence-v1\n[]\n-->', release['body']*2):
            release['body']=body
            with self.assertRaises(ValueError):check('1.0.0',target='linux-x86_64',read=read)

    def test_corruption_cancellation_and_oversize_leave_no_installer(self):
        read, _, _, _, data = self.metadata()
        update = check('1.0.0', target='linux-x86_64', read=read)
        for payload in (b'bad', data + b'extra'):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    stage(update, directory, open_url=lambda _: io.BytesIO(payload))
                self.assertEqual(list(Path(directory).iterdir()), [])
        with tempfile.TemporaryDirectory() as directory:
            cancelled = Event()
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                stage(update, directory, open_url=lambda _: io.BytesIO(data),
                      cancelled=cancelled, progress=lambda _: cancelled.set())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_urls_prereleases_duplicates_and_channel_spoofs_rejected(self):
        for url in ('http://github.com/x', 'https://evil.test/x', 'https://github.com@evil.test/x',
                    'https://github.com:444/x', 'file:///tmp/a'):
            with self.assertRaises(ValueError):
                _url(url)
        read, release, _, _, _ = self.metadata()
        release['prerelease'] = True
        with self.assertRaisesRegex(ValueError, 'stable'):
            check('1.0.0', target='linux-x86_64', read=read)
        release['prerelease'] = False
        release['assets'][0]['browser_download_url'] = 'https://github.com/another/repo/file'
        with self.assertRaisesRegex(ValueError, 'asset'):
            check('1.0.0', target='linux-x86_64', read=read)
