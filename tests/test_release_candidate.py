"""Promotion rejects stale, incomplete, untrusted or changed candidate artifacts."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def release_tool():
    path = Path(__file__).resolve().parents[1] / 'scripts/release-candidate.py'
    spec = importlib.util.spec_from_file_location('release_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tool = release_tool()
        self.identity = self.tool.context('owner/scout', 'a' * 40, 123, 2)
        self.downloaded = self.root / 'downloaded'
        self.downloaded.mkdir()
        for target, (system, arch, _, _) in self.tool.PLATFORMS.items():
            directory = self.directory(target)
            directory.mkdir()
            name = self.tool.installer_name('1.2.3', target)
            (directory / name).write_bytes(f'fixture installer for {target}'.encode())
            (directory / 'build-info.json').write_text(json.dumps(dict(version='1.2.3',
                platform=system, architecture=arch, installer=name)))
            (directory / 'test-installer.py').write_text('# temporary test fixture\n')
            with patch.object(self.tool.metadata, 'distributions', return_value=[]):
                self.tool.create(directory, target, self.identity)
            (directory / 'smoke-report.json').write_text(json.dumps(dict(ok=True, frozen=True,
                version='1.2.3', checks={key: True for key in self.tool.SMOKE_CHECKS})))
            self.tool.record_test(directory, target, self.identity)

    def directory(self, target='linux-x86_64'):
        return self.downloaded / f'tested-installer-{target}'

    def change_json(self, name, change, target='linux-x86_64'):
        path = self.directory(target) / name
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def assemble(self, **kwargs):
        return self.tool.assemble(self.downloaded, kwargs.get('identity', self.identity),
                                  kwargs.get('tag', 'v1.2.3'), self.root / 'release')

    def test_release_contains_exact_tested_installer_bytes_and_evidence(self):
        self.assemble()
        output = self.root / 'release'
        names = {self.tool.installer_name('1.2.3', target) for target in self.tool.PLATFORMS}
        self.assertEqual({path.name for path in output.iterdir()},
                         names | {'SHA256SUMS.txt', 'release-manifest.json'})
        for target in self.tool.PLATFORMS:
            name = self.tool.installer_name('1.2.3', target)
            self.assertEqual((output / name).read_bytes(), (self.directory(target) / name).read_bytes())
            self.assertIn(f'{self.tool.sha256(output / name)}  {name}\n',
                          (output / 'SHA256SUMS.txt').read_text())
        manifest = json.loads((output / 'release-manifest.json').read_text())
        self.assertEqual(manifest['context'], self.identity)
        self.assertEqual(len(manifest['platforms']), 4)

    def test_altered_installer_and_helper_are_rejected(self):
        for filename in ('Scout-Studio-1.2.3-linux-x86_64.deb', 'release-candidate.py',
                         'test-installer.py', 'build-info.json'):
            with self.subTest(file=filename):
                path = self.directory() / filename
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'Candidate file changed'):
                    self.assemble()
                path.write_bytes(original)
        self.assertFalse((self.root / 'release').exists())

    def test_stale_and_cross_build_test_evidence_is_rejected(self):
        for key, value in (('install_and_uninstall_passed', False), ('candidate_sha256', '0' * 64),
                           ('installer_sha256', '0' * 64), ('smoke_sha256', '0' * 64),
                           ('context', dict(self.identity, run_attempt=1))):
            with self.subTest(key=key):
                path = self.directory() / 'test-result.json'
                original = path.read_bytes()
                self.change_json(path.name, lambda data: data.update({key: value}))
                with self.assertRaisesRegex(ValueError, 'install evidence'):
                    self.assemble()
                path.write_bytes(original)

    def test_changed_smoke_report_is_rejected_even_when_still_ok(self):
        self.change_json('smoke-report.json', lambda data: data.update(extra='different report'))
        with self.assertRaisesRegex(ValueError, 'install evidence'):
            self.assemble()

    def test_bad_smoke_check_cannot_be_recorded(self):
        (self.directory() / 'test-result.json').unlink()
        self.change_json('smoke-report.json', lambda data: data['checks'].update(playback=False))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.tool.record_test(self.directory(), 'linux-x86_64', self.identity)
        self.assertFalse((self.directory() / 'test-result.json').exists())

    def test_wrong_frozen_status_version_or_result_cannot_be_recorded(self):
        (self.directory() / 'test-result.json').unlink()
        path = self.directory() / 'smoke-report.json'
        original = path.read_bytes()
        for key, value in (('ok', False), ('frozen', False), ('version', '1.2.4')):
            with self.subTest(key=key):
                self.change_json(path.name, lambda data: data.update({key: value}))
                with self.assertRaisesRegex(ValueError, 'failed or mismatched'):
                    self.tool.record_test(self.directory(), 'linux-x86_64', self.identity)
                path.write_bytes(original)

    def test_missing_platform_or_extra_artifact_cannot_make_a_release(self):
        path = self.directory()
        renamed = path.with_name('unexpected')
        path.rename(renamed)
        with self.assertRaisesRegex(ValueError, 'exactly four'):
            self.assemble()
        renamed.rename(path)
        (self.downloaded / 'unexpected').mkdir()
        with self.assertRaisesRegex(ValueError, 'exactly four'):
            self.assemble()

    def test_tag_platform_commit_and_attempt_mismatches_are_rejected(self):
        for identity, tag in ((self.identity, 'v1.2.4'),
                              (dict(self.identity, commit='b' * 40), 'v1.2.3'),
                              (dict(self.identity, run_attempt=3), 'v1.2.3')):
            with self.subTest(identity=identity, tag=tag), self.assertRaises(ValueError):
                self.assemble(identity=identity, tag=tag)
        self.change_json('candidate.json', lambda data: data.update(platform='windows-x86_64'))
        with self.assertRaisesRegex(ValueError, 'platform mismatch'):
            self.assemble()

    def test_symlink_and_traversal_inputs_are_rejected(self):
        name = self.tool.installer_name('1.2.3', 'linux-x86_64')
        path = self.directory() / name
        destination = self.root / 'outside.deb'
        path.rename(destination)
        try:
            path.symlink_to(destination)
        except OSError:
            self.skipTest('symlink privileges unavailable')
        with self.assertRaisesRegex(ValueError, 'unsafe file'):
            self.assemble()
        for unsafe in ('../outside.deb', '/outside.deb', 'C:\\outside.deb', '..', 'nested/outside'):
            with self.subTest(name=unsafe), self.assertRaisesRegex(ValueError, 'Unsafe artifact filename'):
                self.tool.file_at(self.directory(), unsafe)
        self.assertTrue(destination.is_file())

    def test_existing_release_and_evidence_are_never_replaced(self):
        output = self.root / 'release'
        output.mkdir()
        (output / 'keep').write_bytes(b'previous release')
        with self.assertRaisesRegex(ValueError, 'must not already exist'):
            self.assemble()
        self.assertEqual((output / 'keep').read_bytes(), b'previous release')
        original = (self.directory() / 'test-result.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.tool.record_test(self.directory(), 'linux-x86_64', self.identity)
        self.assertEqual((self.directory() / 'test-result.json').read_bytes(), original)


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tool = release_tool()
        self.repository, self.commit = 'owner/scout', 'a' * 40
        self.workflow = dict(id=99, path=self.tool.WORKFLOW)
        self.run = dict(id=123, run_number=42, run_attempt=2, head_sha=self.commit,
                        head_branch='main', event='push', status='completed', conclusion='success',
                        workflow_id=99, head_repository=dict(full_name=self.repository),
                        repository=dict(full_name=self.repository))
        self.jobs = []
        self.artifacts = []
        for target in self.tool.PLATFORMS:
            smoke = ('Install, exercise and uninstall (Linux)' if target.startswith('linux')
                     else 'Install, exercise and uninstall (Windows/macOS)')
            for name, steps in ((f'Build {target}', ['Build native installer', 'Create candidate manifest']),
                                (f'Install and smoke test {target}',
                                 [smoke, 'Record successful installed-candidate checks', 'Upload tested candidate'])):
                self.jobs.append(dict(name=name, conclusion='success',
                                      steps=[dict(name=step, conclusion='success') for step in steps]))
            self.artifacts.append(dict(name=f'tested-installer-{target}', expired=False))
        self.calls = []

    def api(self, path):
        self.calls.append(path)
        if path.endswith('/workflows/installers.yml'):
            return self.workflow
        if '/workflows/installers.yml/runs?' in path:
            return dict(workflow_runs=[self.run])
        if '/attempts/2/jobs?' in path:
            return dict(jobs=self.jobs)
        if '/runs/123/artifacts?' in path:
            return dict(artifacts=self.artifacts)
        self.fail(f'Unexpected API path: {path}')

    def select(self):
        return self.tool.select_candidate(self.api, self.repository, self.commit)

    def test_exact_trusted_successful_run_and_attempt_selected(self):
        self.assertEqual(self.select(), self.tool.context(self.repository, self.commit, 123, 2))
        self.assertTrue(any('/attempts/2/jobs?' in path for path in self.calls))

    def test_fork_pr_wrong_branch_commit_or_workflow_cannot_be_promoted(self):
        for key, value in (('event', 'pull_request'), ('head_branch', 'feature'),
                           ('head_sha', 'b' * 40), ('workflow_id', 100),
                           ('status', 'in_progress'), ('conclusion', 'failure'),
                           ('head_repository', dict(full_name='fork/scout')),
                           ('repository', dict(full_name='fork/scout'))):
            with self.subTest(key=key):
                original = copy.deepcopy(self.run)
                self.run[key] = value
                with self.assertRaisesRegex(ValueError, 'No successful main-branch'):
                    self.select()
                self.run = original

    def test_failed_missing_duplicate_or_skipped_jobs_block_promotion(self):
        original = copy.deepcopy(self.jobs)
        for modification in ('failure', 'skipped', 'missing', 'duplicate'):
            with self.subTest(modification=modification):
                self.jobs = copy.deepcopy(original)
                if modification == 'missing':
                    self.jobs.pop()
                elif modification == 'duplicate':
                    self.jobs.append(self.jobs[-1])
                else:
                    self.jobs[-1]['conclusion'] = modification
                with self.assertRaisesRegex(ValueError, 'current-attempt job'):
                    self.select()

    def test_missing_or_skipped_actual_installer_step_blocks_promotion(self):
        for conclusion in ('skipped', 'failure'):
            with self.subTest(conclusion=conclusion):
                self.jobs[1]['steps'][0]['conclusion'] = conclusion
                with self.assertRaisesRegex(ValueError, 'required step'):
                    self.select()

    def test_expired_missing_or_duplicate_artifacts_block_promotion(self):
        original = copy.deepcopy(self.artifacts)
        for modification in ('expired', 'missing', 'duplicate'):
            with self.subTest(modification=modification):
                self.artifacts = copy.deepcopy(original)
                if modification == 'expired':
                    self.artifacts[-1]['expired'] = True
                elif modification == 'missing':
                    self.artifacts.pop()
                else:
                    self.artifacts.append(self.artifacts[-1])
                with self.assertRaisesRegex(ValueError, 'Missing/expired/duplicate'):
                    self.select()


if __name__ == '__main__':
    unittest.main()

class UpgradePromotionTests(unittest.TestCase):
    setUp = CandidateTests.setUp
    directory = CandidateTests.directory
    assemble = CandidateTests.assemble

    def test_declared_inference_profile_requires_matching_installed_smoke(self):
        directory = self.directory()
        for name in ('candidate.json', 'release-candidate.py', 'test-result.json'):
            (directory / name).unlink()
        path = directory / 'build-info.json'
        info = json.loads(path.read_text())
        info['runtime_profile'] = 'inference'
        path.write_text(json.dumps(info))
        with patch.object(self.tool.metadata, 'distributions', return_value=[]):
            self.tool.create(directory, 'linux-x86_64', self.identity)
        with self.assertRaisesRegex(ValueError, 'runtime profile'):
            self.tool.record_test(directory, 'linux-x86_64', self.identity)
        report_path = directory / 'smoke-report.json'
        report = json.loads(report_path.read_text())
        report['checks']['runtime_profile'] = 'inference'
        report_path.write_text(json.dumps(report))
        self.tool.record_test(directory, 'linux-x86_64', self.identity)

    def test_required_upgrade_blocks_bootstrap_unless_explicitly_allowed(self):
        with self.assertRaisesRegex(ValueError, 'requires a verified'):
            self.tool.assemble(self.downloaded, self.identity, 'v1.2.3', self.root / 'release', require_upgrade=True)
        self.assertFalse((self.root / 'release').exists())

    def test_upgrade_hashes_and_post_uninstall_evidence_are_required(self):
        target = 'linux-x86_64'
        directory = self.directory(target)
        for name in ('candidate.json', 'release-candidate.py', 'test-result.json'):
            (directory / name).unlink()
        info_path = directory / 'build-info.json'
        info = json.loads(info_path.read_text())
        old = self.tool.installer_name('1.0.0', target)
        (directory / old).write_bytes(b'previous release fixture')
        digest = self.tool.sha256(directory / old)
        info['upgrade_baseline'] = dict(version='1.0.0', installer=old, sha256=digest, size=(directory / old).stat().st_size)
        info_path.write_text(json.dumps(info))
        (directory / 'upgrade-baseline.json').write_text(json.dumps(dict(version='1.0.0', installer=old,
                    platform='Linux', architecture='x86_64')))
        (directory / (old + '.sha256')).write_text(digest + '  ' + old)
        with patch.object(self.tool.metadata, 'distributions', return_value=[]):
            manifest = self.tool.create(directory, target, self.identity)
        with self.assertRaises(ValueError):
            self.tool.record_test(directory, target, self.identity)
        report = dict(schema=1, previous_version='1.0.0', version='1.2.3', previous_installer_sha256=digest,
                      installer_sha256=manifest['files'][manifest['installer']]['sha256'],
                      protected_files={'review.mlt': 'c' * 64}, upgrade_passed=True, uninstall_preserved_data=False)
        report_path = directory / 'upgrade-report.json'
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'Upgrade evidence'):
            self.tool.record_test(directory, target, self.identity)
        report['uninstall_preserved_data'] = True
        report_path.write_text(json.dumps(report))
        result = self.tool.record_test(directory, target, self.identity)
        self.assertEqual(result['upgrade_sha256'], self.tool.sha256(report_path))
        report['protected_files'] = {}
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'Upgrade evidence'):
            self.assemble()
