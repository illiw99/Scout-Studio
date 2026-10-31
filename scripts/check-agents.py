#!/usr/bin/env python3
"""Validate local agent configuration offline (3.10: optional tomli needed)."""
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise SystemExit(
            'Agent validation needs Python 3.11+ or optional tomli on Python 3.10. '
            'Install it with: python3 -m pip install tomli'
        ) from None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / '.codex/config.toml').read_text(encoding='utf-8'))
    require(set(config) == {'agents'}, 'Project config must inherit model and permissions')
    agents = config['agents']
    require(set(agents) == {'enabled', 'max_concurrent_threads_per_session'},
            'Unexpected agent configuration keys')
    require(agents['enabled'] is True, 'Subagents must be enabled')
    require(type(agents['max_concurrent_threads_per_session']) is int
            and agents['max_concurrent_threads_per_session'] == 2,
            'Expected two concurrent child threads')
    expected = {'researcher', 'pipeline', 'desktop', 'infrastructure', 'verifier', 'reviewer'}
    names = set()
    for path in sorted((root / '.codex/agents').glob('*.toml')):
        role = tomllib.loads(path.read_text(encoding='utf-8'))
        require(set(role) == {'name', 'description', 'developer_instructions'},
                f'{path}: unexpected role keys; inherit model and permissions')
        for key in ('name', 'description', 'developer_instructions'):
            require(isinstance(role.get(key), str) and role[key].strip(),
                    f'{path}: missing {key}')
        require(role['name'] == path.stem, f'{path}: filename/name mismatch')
        require(role['name'] not in names, f'{path}: duplicate role')
        require('Do not spawn subagents.' in role['developer_instructions'],
                f'{path}: missing no-nesting instruction')
        names.add(role['name'])
    require(names == expected, f'Unexpected roles: {names ^ expected}')
    for artifact in ('AGENTS.md', 'docs/code-map.md', 'docs/multi-agent-workflow.md'):
        require((root / artifact).is_file(), f'Missing {artifact}')
    print(f'Agent setup valid: {len(names)} roles, 2 concurrent child threads.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'Agent validation failed: {error}', file=sys.stderr)
        sys.exit(1)
