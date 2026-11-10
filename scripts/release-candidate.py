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
MAX_RELEASE_ASSET = 2 * 1024 ** 3


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


def read_json(path, maximum=2 * 1024 * 1024):
    require(not path.is_symlink() and path.is_file(), f'Missing or unsafe JSON: {path.name}')
    require(path.stat().st_size <= maximum, 'Oversized manifest/report')
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


def baseline_files(info, target):
    baseline = info.get('upgrade_baseline')
    if baseline is None:
        return ()
    require(isinstance(baseline, dict), 'Invalid upgrade baseline')
    name = installer_name(baseline.get('version'), target)
    require(name == baseline.get('installer') and name != info.get('installer')
            and isinstance(baseline.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', baseline['sha256'])
            and type(baseline.get('size')) is int and baseline['size'] > 0,
            'Invalid upgrade baseline identity')
    require(tuple(map(int, baseline['version'].split('.'))) < tuple(map(int, info['version'].split('.'))),
            'Upgrade baseline must be older than candidate')
    return (name, name + '.sha256', 'upgrade-baseline.json')


def processing_files(directory, info, target):
    """Bind every separately downloaded engine part to the tested installer."""
    name = info.get('processing_catalog')
    if name is None:
        require(info.get('runtime_profile') != 'bootstrap', 'Bootstrap installer has no engine catalogue')
        return ()
    require(name == 'runtime-catalog.json', 'Unexpected engine catalogue name')
    catalog = read_json(file_at(directory, name), maximum=8 * 1024 * 1024)
    expected = {target} if target.startswith('macos-') else {target + '-cpu', target + '-cuda'}
    require(catalog.get('schema') == 1 and catalog.get('application_version') == info['version']
            and isinstance(catalog.get('runtimes'), dict) and set(catalog['runtimes']) == expected,
            'Incompatible processing-engine catalogue')
    names = [name]
    for engine_id, spec in catalog['runtimes'].items():
        parts = spec.get('parts') if isinstance(spec, dict) else None
        require(isinstance(parts, list) and 1 <= len(parts) <= 32, 'Invalid engine part set')
        for index, part in enumerate(parts, 1):
            filename = f'Scout-Runtime-{info["version"]}-{engine_id}.tar.gz.part{index:03d}'
            require(isinstance(part, dict) and part.get('name') == filename
                    and type(part.get('size')) is int and 0 < part['size'] <= 1024 ** 3
                    and isinstance(part.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', part['sha256']),
                    'Invalid engine part identity')
            path = file_at(directory, filename)
            require(path.stat().st_size == part['size'] and sha256(path) == part['sha256'],
                    'Processing-engine part changed: ' + filename)
            names.append(filename)
    require(len(set(names)) == len(names), 'Duplicate engine part')
    return tuple(names)


def processing_smoke(directory, manifest, report):
    if manifest.get('runtime_profile') != 'bootstrap':
        return
    info = read_json(directory / 'build-info.json')
    catalog_name = info.get('processing_catalog')
    checks = report.get('checks', {})
    require(checks.get('runtime_setup') is True
            and checks.get('runtime_catalog_sha256') == manifest['files'][catalog_name]['sha256'],
            'Installed catalogue differs from the tested candidate')
    catalog = read_json(directory / catalog_name, maximum=8 * 1024 * 1024)
    engines = checks.get('processing_engines')
    require(isinstance(engines, dict) and set(engines) == set(catalog['runtimes']),
            'Every native engine must pass an installed probe')
    for engine_id, probe in engines.items():
        require(isinstance(probe, dict) and probe.get('ok') is True
                and probe.get('nms') is True and probe.get('inference') is True
                and probe.get('application_version') == manifest['version']
                and probe.get('engine_id') == engine_id and 'cpu' in probe.get('devices', []),
                'Processing-engine probe failed or mismatched')
        if engine_id.endswith('-cuda'):
            require(isinstance(probe.get('cuda_build'), str) and probe['cuda_build'],
                    'CUDA engine lost CUDA build support')


def create(directory, target, identity):
    info = read_json(directory / 'build-info.json')
    name = installer_name(info.get('version'), target)
    system, arch, _, minimum = PLATFORMS[target]
    actual_arch = str(info.get('architecture', '')).lower().replace('amd64', 'x86_64')
    require((info.get('platform'), actual_arch, info.get('installer')) == (system, arch, name),
            'Build metadata does not match target installer')
    helper = directory / 'release-candidate.py'
    require(not helper.exists() and not helper.is_symlink(), 'Candidate helper already exists')
    processing = processing_files(directory, info, target)
    for filename in (name, *processing):
        require(file_at(directory, filename).stat().st_size < MAX_RELEASE_ASSET,
                'Release asset exceeds GitHub size limit: ' + filename)
    shutil.copyfile(Path(__file__), helper)
    names = (name, 'build-info.json', 'test-installer.py', helper.name) + baseline_files(info, target) + processing
    files = {item: dict(sha256=sha256(file_at(directory, item)),
                        size=file_at(directory, item).stat().st_size) for item in names}
    result = dict(schema=1, context=identity, platform=target, version=info['version'],
                  installer=name, files=files, minimum_supported_os=minimum,
                  build_os=platform.platform(), python=platform.python_version(),
                  dependencies=sorted({(dist.metadata['Name'], dist.version)
                                       for dist in metadata.distributions() if dist.metadata['Name']}),
                  model_packs=[], runtime_profile=info.get('runtime_profile', 'reference'), signing=info.get('signing', 'not-configured'))
    write_new(directory / 'candidate.json', result)
    return result


def verify(directory, target, identity):
    manifest = read_json(directory / 'candidate.json')
    require(manifest.get('schema') == 1 and manifest.get('context') == identity,
            'Candidate build identity mismatch')
    require(manifest.get('platform') == target, 'Candidate platform mismatch')
    name = installer_name(manifest.get('version'), target)
    require(manifest.get('installer') == name, 'Unexpected installer name')
    build_record = manifest.get('files', {}).get('build-info.json', {})
    require(build_record.get('sha256') == sha256(file_at(directory, 'build-info.json')), 'Candidate file changed: build-info.json')
    info = read_json(directory / 'build-info.json')
    expected = {name, 'build-info.json', 'test-installer.py', 'release-candidate.py',
                *baseline_files(info, target), *processing_files(directory, info, target)}
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
    require(info.get('runtime_profile', 'reference') in ('reference', 'inference', 'bootstrap')
            and manifest.get('runtime_profile', 'reference') == info.get('runtime_profile', 'reference'),
            'Candidate runtime profile mismatch')
    baseline = info.get('upgrade_baseline')
    if baseline:
        previous = read_json(directory / 'upgrade-baseline.json')
        require(previous == {key: (baseline[key] if key in ('installer', 'version') else info[key])
                             for key in ('version', 'installer', 'platform', 'architecture')}, 'Upgrade baseline metadata mismatch')
        require(sha256(file_at(directory, baseline['installer'])) == baseline['sha256']
                and file_at(directory, baseline['installer']).stat().st_size == baseline['size'], 'Upgrade baseline bytes mismatch')
    require(manifest.get('signing') == info.get('signing', 'not-configured'), 'Candidate signing metadata mismatch')
    return manifest


def smoke(directory, version):
    report = read_json(directory / 'smoke-report.json')
    require(report.get('ok') is True and report.get('frozen') is True
            and report.get('version') == version, 'Installed-app smoke report failed or mismatched')
    checks = report.get('checks')
    require(isinstance(checks, dict) and all(checks.get(key) is True for key in SMOKE_CHECKS),
            'Installed-app smoke checks are incomplete')
    info = read_json(directory / 'build-info.json')
    if 'runtime_profile' in info:
        require(checks.get('runtime_profile') == info['runtime_profile'], 'Installed runtime profile mismatch')
    return report


def upgrade_evidence(directory, manifest, *, required=False):
    info = read_json(directory / 'build-info.json')
    baseline = info.get('upgrade_baseline')
    if not baseline:
        require(not required, 'Promotion requires a verified previous-release upgrade; configure a baseline or explicitly permit the first release')
        return {}
    report = read_json(directory / 'upgrade-report.json')
    require(report.get('schema') == 1 and report.get('upgrade_passed') is True
            and report.get('uninstall_preserved_data') is True
            and report.get('previous_version') == baseline['version'] and report.get('version') == manifest['version']
            and report.get('previous_installer_sha256') == baseline['sha256']
            and report.get('installer_sha256') == manifest['files'][manifest['installer']]['sha256']
            and isinstance(report.get('protected_files'), dict) and bool(report['protected_files']),
            'Upgrade evidence is missing, failed or mismatched')
    return dict(upgrade_sha256=sha256(directory / 'upgrade-report.json'))


def record_test(directory, target, identity):
    manifest = verify(directory, target, identity)
    report = smoke(directory, manifest['version'])
    processing_smoke(directory, manifest, report)
    result = dict(schema=1, context=identity, platform=target,
                  candidate_sha256=sha256(directory / 'candidate.json'),
                  installer_sha256=manifest['files'][manifest['installer']]['sha256'],
                  smoke_sha256=sha256(directory / 'smoke-report.json'),
                  install_and_uninstall_passed=True)
    result.update(upgrade_evidence(directory, manifest))
    write_new(directory / 'test-result.json', result)
    return result


def assemble(downloaded, selection, tag, output, *, require_upgrade=False):
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
        expected_upgrade = upgrade_evidence(directory, manifest, required=require_upgrade)
        require(evidence == dict(**expected_upgrade, schema=1, context=identity, platform=target,
                    candidate_sha256=sha256(directory / 'candidate.json'),
                    installer_sha256=manifest['files'][manifest['installer']]['sha256'],
                    smoke_sha256=sha256(file_at(directory, 'smoke-report.json')),
                    install_and_uninstall_passed=True), 'Missing, stale or mismatched install evidence')
        report = smoke(directory, manifest['version'])
        processing_smoke(directory, manifest, report)
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
            info = read_json(directory / 'build-info.json')
            for part_name in processing_files(directory, info, manifest['platform']):
                if part_name == 'runtime-catalog.json':
                    continue  # Each installer embeds its pinned platform catalogue.
                require(not (stage / part_name).exists(), 'Duplicate published engine part')
                shutil.copyfile(file_at(directory, part_name), stage / part_name)
                part_digest = sha256(stage / part_name)
                require(part_digest == manifest['files'][part_name]['sha256']
                        and (stage / part_name).stat().st_size < MAX_RELEASE_ASSET,
                        'Engine part changed during release assembly')
                checksums.append(f'{part_digest}  {part_name}\n')
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
    sub.add_argument('--require-upgrade', action='store_true')
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
            assemble(args.directory, read_json(args.selection), args.tag, args.output, require_upgrade=args.require_upgrade)
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
