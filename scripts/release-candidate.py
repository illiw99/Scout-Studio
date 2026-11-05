#!/usr/bin/env python3
"""Bind installers to their build and installed-app checks; never build or publish.

Only the promotion workflow selects a trusted successful main-branch build via
GitHub's read-only API. Downloaded installers and helper scripts are verified as
data, never executed by promotion. Local checks use temporary fixture artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

WORKFLOW = '.github/workflows/installers.yml'
PLATFORMS = {
    'linux-x86_64': ('Linux', 'x86_64', '.deb', 'Ubuntu 22.04 / glibc 2.35'),
    'windows-x86_64': ('Windows', 'x86_64', '.exe', None),
    'macos-arm64': ('Darwin', 'arm64', '.dmg', 'macOS 15'),
    'macos-x86_64': ('Darwin', 'x86_64', '.dmg', 'macOS 15'),
}
SMOKE_CHECKS = ('dependencies', 'ffmpeg', 'export', 'gui', 'playback', 'cli', 'worker', 'inference')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_at(directory, name):
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9._-]+', name)
            and name not in {'.', '..'}, 'Unsafe artifact filename')
    path = directory / name
    require(not path.is_symlink() and path.is_file(), f'Missing or unsafe file: {name}')
    return path


def read_json(path):
    require(not path.is_symlink() and path.is_file(), f'Missing or unsafe JSON: {path.name}')
    require(path.stat().st_size <= 2 * 1024 * 1024, 'Oversized manifest/report')
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'Expected JSON object: {path.name}')
    return value


def write_new(path, value):
    # New files only. Failed/repeated commands cannot replace prior evidence.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def context(repository, commit, run_id, run_attempt):
    require(isinstance(repository, str) and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository),
            'Invalid repository')
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit), 'Invalid commit SHA')
    require(type(run_id) is int and run_id > 0 and type(run_attempt) is int and run_attempt > 0,
            'Build run ID and attempt must be positive integers')
    return dict(repository=repository, commit=commit, run_id=run_id,
                run_attempt=run_attempt, workflow=WORKFLOW)


def installer_name(version, target):
    require(isinstance(version, str) and re.fullmatch(r'\d+\.\d+\.\d+', version), 'Invalid release version')
    require(target in PLATFORMS, 'Unsupported target platform')
    return f'Scout-Studio-{version}-{target}{PLATFORMS[target][2]}'


def create(directory, target, identity):
    info = read_json(directory / 'build-info.json')
    name = installer_name(info.get('version'), target)
    system, arch, _, minimum = PLATFORMS[target]
    actual_arch = str(info.get('architecture', '')).lower().replace('amd64', 'x86_64')
    require((info.get('platform'), actual_arch, info.get('installer')) == (system, arch, name),
            'Build metadata does not match target installer')
    helper = directory / 'release-candidate.py'
    require(not helper.exists() and not helper.is_symlink(), 'Candidate helper already exists')
    shutil.copyfile(Path(__file__), helper)
    names = (name, 'build-info.json', 'test-installer.py', helper.name)
    files = {item: dict(sha256=sha256(file_at(directory, item)),
                        size=file_at(directory, item).stat().st_size) for item in names}
    result = dict(schema=1, context=identity, platform=target, version=info['version'],
                  installer=name, files=files, minimum_supported_os=minimum,
                  build_os=platform.platform(), python=platform.python_version(),
                  dependencies=sorted({(dist.metadata['Name'], dist.version)
                                       for dist in metadata.distributions() if dist.metadata['Name']}),
                  model_packs=[], signing='not-configured')
    write_new(directory / 'candidate.json', result)
    return result


def verify(directory, target, identity):
    manifest = read_json(directory / 'candidate.json')
    require(manifest.get('schema') == 1 and manifest.get('context') == identity,
            'Candidate build identity mismatch')
    require(manifest.get('platform') == target, 'Candidate platform mismatch')
    name = installer_name(manifest.get('version'), target)
    require(manifest.get('installer') == name, 'Unexpected installer name')
    expected = {name, 'build-info.json', 'test-installer.py', 'release-candidate.py'}
    files = manifest.get('files')
    require(isinstance(files, dict) and set(files) == expected, 'Invalid candidate file set')
    for filename, record in files.items():
        path = file_at(directory, filename)
        require(isinstance(record, dict) and type(record.get('size')) is int
                and record['size'] > 0 and record.get('sha256') == sha256(path)
                and record['size'] == path.stat().st_size, f'Candidate file changed: {filename}')
    info = read_json(directory / 'build-info.json')
    system, arch, _, _ = PLATFORMS[target]
    require(info.get('version') == manifest['version'] and info.get('installer') == name
            and info.get('platform') == system
            and str(info.get('architecture', '')).lower().replace('amd64', 'x86_64') == arch,
            'Candidate build metadata mismatch')
    return manifest


def smoke(directory, version):
    report = read_json(directory / 'smoke-report.json')
    require(report.get('ok') is True and report.get('frozen') is True
            and report.get('version') == version, 'Installed-app smoke report failed or mismatched')
    checks = report.get('checks')
    require(isinstance(checks, dict) and all(checks.get(key) is True for key in SMOKE_CHECKS),
            'Installed-app smoke checks are incomplete')
    return report


def record_test(directory, target, identity):
    manifest = verify(directory, target, identity)
    smoke(directory, manifest['version'])
    result = dict(schema=1, context=identity, platform=target,
                  candidate_sha256=sha256(directory / 'candidate.json'),
                  installer_sha256=manifest['files'][manifest['installer']]['sha256'],
                  smoke_sha256=sha256(directory / 'smoke-report.json'),
                  install_and_uninstall_passed=True)
    write_new(directory / 'test-result.json', result)
    return result


def assemble(downloaded, selection, tag, output):
    identity = context(**{key: selection.get(key) for key in
                         ('repository', 'commit', 'run_id', 'run_attempt')})
    require(selection.get('workflow') == WORKFLOW, 'Selected workflow mismatch')
    require(isinstance(tag, str) and re.fullmatch(r'v\d+\.\d+\.\d+', tag), 'Invalid release tag')
    require(not output.exists() and not output.is_symlink(), 'Release output must not already exist')
    expected = {f'tested-installer-{target}' for target in PLATFORMS}
    require({p.name for p in downloaded.iterdir()} == expected, 'Expected exactly four tested platform artifacts')
    candidates = []
    for target in PLATFORMS:
        directory = downloaded / f'tested-installer-{target}'
        require(not directory.is_symlink() and directory.is_dir(), 'Unsafe artifact directory')
        manifest = verify(directory, target, identity)
        require(tag == f"v{manifest['version']}", 'Release tag/version mismatch')
        evidence = read_json(directory / 'test-result.json')
        require(evidence == dict(schema=1, context=identity, platform=target,
                    candidate_sha256=sha256(directory / 'candidate.json'),
                    installer_sha256=manifest['files'][manifest['installer']]['sha256'],
                    smoke_sha256=sha256(file_at(directory, 'smoke-report.json')),
                    install_and_uninstall_passed=True), 'Missing, stale or mismatched install evidence')
        report = smoke(directory, manifest['version'])
        candidates.append((directory, manifest, evidence, report))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Verify all inputs before assembling; expose only a complete release set.
    with tempfile.TemporaryDirectory(prefix='.scout-release-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'release'
        stage.mkdir()
        checksums, platforms = [], []
        for directory, manifest, evidence, report in candidates:
            name = manifest['installer']
            shutil.copyfile(file_at(directory, name), stage / name)
            digest = sha256(stage / name)
            require(digest == evidence['installer_sha256'], 'Installer changed during release assembly')
            checksums.append(f'{digest}  {name}\n')
            platforms.append(dict(candidate=manifest, test_result=evidence, smoke_report=report))
        (stage / 'SHA256SUMS.txt').write_text(''.join(sorted(checksums)), encoding='utf-8')
        write_new(stage / 'release-manifest.json', dict(schema=1, context=identity, tag=tag,
                                                       platforms=platforms))
        stage.rename(output)


def select_candidate(api, repository, commit):
    """Select one complete trusted build; never substitute another commit/build."""
    # Validate before interpolating API routes or using external metadata.
    context(repository, commit, 1, 1)
    base = f'/repos/{repository}/actions'
    workflow = api(f'{base}/workflows/installers.yml')
    require(type(workflow.get('id')) is int and workflow.get('path') == WORKFLOW,
            'Unexpected installer workflow')
    runs = api(f'{base}/workflows/installers.yml/runs?head_sha={commit}&branch=main&event=push&status=success&per_page=100')
    eligible = [run for run in runs.get('workflow_runs', []) if
                run.get('head_sha') == commit and run.get('head_branch') == 'main'
                and run.get('event') == 'push' and run.get('status') == 'completed'
                and run.get('conclusion') == 'success' and run.get('workflow_id') == workflow['id']
                and run.get('head_repository', {}).get('full_name') == repository
                and run.get('repository', {}).get('full_name') == repository]
    require(eligible, 'No successful main-branch installer candidate for this exact commit; build/test it first')
    run = max(eligible, key=lambda item: (item['run_number'], item['run_attempt']))
    identity = context(repository, commit, run['id'], run['run_attempt'])

    def pages(path, key):
        records = []
        for page in range(1, 101):
            response = api(f'{path}?per_page=100&page={page}')
            batch = response[key]
            records.extend(batch)
            if len(batch) < 100:
                return records
        raise ValueError('Too many workflow records to validate')

    jobs = pages(f"{base}/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", 'jobs')
    for target in PLATFORMS:
        smoke_step = ('Install, exercise and uninstall (Linux)' if target.startswith('linux')
                      else 'Install, exercise and uninstall (Windows/macOS)')
        for name, steps in ((f'Build {target}', ('Build native installer', 'Create candidate manifest')),
                            (f'Install and smoke test {target}',
                             (smoke_step, 'Record successful installed-candidate checks', 'Upload tested candidate'))):
            matches = [job for job in jobs if job.get('name') == name]
            require(len(matches) == 1 and matches[0].get('conclusion') == 'success',
                    f'Missing/failed current-attempt job: {name}')
            for step in steps:
                matches_steps = [item for item in matches[0].get('steps', []) if item.get('name') == step]
                require(len(matches_steps) == 1 and matches_steps[0].get('conclusion') == 'success',
                        f'Missing/failed required step: {step}')
    artifacts = pages(f"{base}/runs/{run['id']}/artifacts", 'artifacts')
    for target in PLATFORMS:
        matches = [item for item in artifacts if item.get('name') == f'tested-installer-{target}']
        require(len(matches) == 1 and matches[0].get('expired') is False,
                f'Missing/expired/duplicate tested artifact for {target}; rebuild and retest')
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('create', 'verify', 'record-test'):
        sub = commands.add_parser(name)
        sub.add_argument('--directory', type=Path, required=True)
        sub.add_argument('--platform', choices=PLATFORMS, required=True)
        sub.add_argument('--repository', required=True)
        sub.add_argument('--commit', required=True)
        sub.add_argument('--run-id', type=int, required=True)
        sub.add_argument('--run-attempt', type=int, required=True)
    sub = commands.add_parser('select')
    sub.add_argument('--repository', required=True)
    sub.add_argument('--commit', required=True)
    sub.add_argument('--output', type=Path, required=True)
    sub = commands.add_parser('assemble')
    sub.add_argument('--directory', type=Path, required=True)
    sub.add_argument('--selection', type=Path, required=True)
    sub.add_argument('--tag', required=True)
    sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in {'create', 'verify', 'record-test'}:
            identity = context(args.repository, args.commit, args.run_id, args.run_attempt)
            function = {'create': create, 'verify': verify, 'record-test': record_test}[args.command]
            function(args.directory, args.platform, identity)
        elif args.command == 'assemble':
            assemble(args.directory, read_json(args.selection), args.tag, args.output)
        else:
            token = os.environ.get('GH_TOKEN')
            require(token, 'GH_TOKEN is required to select a trusted candidate')

            def api(path):
                request = Request('https://api.github.com' + path, headers={
                    'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28'})
                with urlopen(request, timeout=30) as response:
                    return json.load(response)

            identity = select_candidate(api, args.repository, args.commit)
            write_new(args.output, identity)
            if os.environ.get('GITHUB_OUTPUT'):
                with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
                    stream.write(f"run_id={identity['run_id']}\n")
            print(json.dumps(identity, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Candidate verification failed: {exc}\n')


if __name__ == '__main__':
    main()
