#!/usr/bin/env python3
"""Preview or explicitly migrate game/source state folders to Scout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scout.state_migration import apply_migration, plan_migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path, help='game/source directories containing .scout or scout')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='preview only (default)')
    mode.add_argument('--apply', action='store_true', help='apply after Studio and media jobs are closed')
    parser.add_argument('--backup-dir', type=Path, help='new backup directory; required with --apply')
    args = parser.parse_args()
    if args.apply and args.backup_dir is None:
        parser.error('--apply requires --backup-dir')
    try:
        plan = plan_migration(args.folders)
        result = apply_migration(plan, args.backup_dir) if args.apply else plan
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Migration refused: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
