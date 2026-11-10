"""Runtime archives must exactly describe regular, reproducible native files."""
from __future__ import annotations
import importlib.util
import io
import json
import os
import shutil
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runtime = module('build_runtime', 'scripts/build-runtime.py')
installer = module('build_installers_runtime', 'scripts/build-installers.py')
launcher = module('worker_launcher', 'packaging/worker_launcher.py')


class RuntimeBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / 'frozen'
        self.bundle.mkdir()
        executable = self.bundle / 'Scout-Worker'
        executable.write_bytes(b'#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
        (self.bundle / 'library.dat').write_bytes(bytes(range(256)) * 3)

    def pack(self, engine='linux-x86_64-cpu', output=None):
        return runtime.pack_runtime(self.bundle, output or self.root / 'output', engine, '1.0.2', part_size=100)

    def test_parts_and_members_exactly_match_archive(self):
        document = self.pack()
        spec = document['runtimes']['linux-x86_64-cpu']
        self.assertGreater(len(spec['parts']), 1)
        joined = b''
        for part in spec['parts']:
            path = self.root / 'output' / part['name']
            self.assertLessEqual(part['size'], 100)
            self.assertEqual(path.stat().st_size, part['size'])
            self.assertEqual(runtime.sha256(path), part['sha256'])
            joined += path.read_bytes()
        with tarfile.open(fileobj=io.BytesIO(joined), mode='r:gz') as archive:
            self.assertEqual(set(archive.getnames()), set(spec['members']))
            for member in archive:
                self.assertTrue(member.isfile())
                self.assertEqual(member.mode, spec['members'][member.name]['mode'])
                self.assertEqual(member.size, spec['members'][member.name]['size'])
                data = archive.extractfile(member).read()
                self.assertEqual(runtime.hashlib.sha256(data).hexdigest(), spec['members'][member.name]['sha256'])
        self.assertEqual(spec['uncompressed_size'], sum(item['size'] for item in spec['members'].values()))
        self.assertEqual(spec['members'][spec['entrypoint']]['mode'], 0o755)

    def test_pack_is_reproducible_and_source_is_preserved(self):
        first = self.pack()
        second = self.pack(output=self.root / 'second')
        self.assertEqual(first, second)
        self.assertEqual((self.bundle / 'library.dat').read_bytes(), bytes(range(256))*3)

    def test_windows_writable_stat_mode_is_normalized_in_archive_metadata(self):
        (self.bundle / 'library.dat').chmod(0o666)
        # Windows retains 0666 after chmod(0644); preserve that condition in the
        # staging fixture even on POSIX so the catalogue contract is exercised.
        with patch.object(runtime, 'materialize', side_effect=shutil.copytree):
            spec = self.pack()['runtimes']['linux-x86_64-cpu']
        self.assertEqual(spec['members']['Scout-Worker/library.dat']['mode'], 0o644)

    @unittest.skipIf(os.name == 'nt', 'Windows symlinks require privileges')
    def test_library_links_are_materialized_as_regular_archive_members(self):
        (self.bundle / 'alias.dat').symlink_to('library.dat')
        spec = self.pack()['runtimes']['linux-x86_64-cpu']
        self.assertEqual(spec['members']['Scout-Worker/alias.dat'], spec['members']['Scout-Worker/library.dat'])
        self.assertTrue((self.bundle / 'alias.dat').is_symlink())

    @unittest.skipIf(os.name == 'nt', 'Windows symlinks require privileges')
    def test_cyclic_links_fail_without_published_metadata(self):
        (self.bundle / 'cycle').symlink_to('.', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            self.pack()
        self.assertFalse((self.root / 'output/runtime-linux-x86_64-cpu.json').exists())

    def test_missing_entrypoint_and_oversized_parts_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.pack('windows-x86_64-cpu')
        with self.assertRaisesRegex(ValueError, 'part size'):
            runtime.pack_runtime(self.bundle, self.root / 'output', 'linux-x86_64-cpu', '1.0.2', part_size=runtime.PART_SIZE+1)

    def test_catalog_requires_host_engines_and_validates_parts(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, 'Missing native'):
            installer.prepare_processing_catalog(self.root / 'output', self.root / 'output', '1.0.2', 'Linux', 'x86_64')
        self.pack('linux-x86_64-cuda')
        catalog = installer.prepare_processing_catalog(self.root / 'output', self.root / 'output', '1.0.2', 'Linux', 'x86_64')
        self.assertEqual(set(json.loads(catalog.read_text())['runtimes']), {'linux-x86_64-cpu', 'linux-x86_64-cuda'})
        part = next((self.root / 'output').glob('*cpu*.part001'))
        part.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            installer.prepare_processing_catalog(self.root / 'output', self.root / 'output', '1.0.2', 'Linux', 'x86_64')

    def test_catalog_rejects_incompatible_version_and_unsafe_members(self):
        self.pack()
        self.pack('linux-x86_64-cuda')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            installer.prepare_processing_catalog(self.root / 'output', self.root / 'output', '1.0.3', 'Linux', 'x86_64')
        path = self.root / 'output/runtime-linux-x86_64-cpu.json'
        data = json.loads(path.read_text())
        spec = data['runtimes']['linux-x86_64-cpu']
        spec['members']['Scout-Worker/../escape'] = next(iter(spec['members'].values()))
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'member'):
            installer.prepare_processing_catalog(self.root / 'output', self.root / 'output', '1.0.2', 'Linux', 'x86_64')

    def test_worker_routes_existing_protocol_without_desktop_import(self):
        worker, cli = SimpleNamespace(main=lambda argv: ('worker', argv)), SimpleNamespace(main=lambda argv: ('cli', argv))
        with patch.dict('sys.modules', {'scout.worker': worker, 'scout.cli': cli}):
            self.assertEqual(launcher.main(['--worker-job', '/tmp/job']), ('worker', ['/tmp/job']))
            self.assertEqual(launcher.main(['--scout-cli', '--help']), ('cli', ['--help']))
        with self.assertRaises(SystemExit):
            launcher.main(['--unknown'])

    def test_failed_probe_writes_machine_readable_report(self):
        report = self.root / 'probe.json'
        with patch.dict('sys.modules', {'torch': None}):
            self.assertEqual(launcher.engine_probe(report), 1)
        data = json.loads(report.read_text())
        self.assertFalse(data['ok'])
        self.assertFalse(data['nms'])
        self.assertIn('application_version', data)
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
