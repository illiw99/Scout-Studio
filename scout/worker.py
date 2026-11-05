"""Versioned local job protocol. Heavy processing imports occur after startup."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

SCHEMA = 1
OPTIONS = {
    'run': {}, 'detect': {}, 'segment': {},
    'cut': {'name_pattern': str, 'start_number': int, 'number_width': int, 'review_hash': str},
}


def validate_job(data):
    if not isinstance(data, dict) or set(data) != {'schema', 'job_id', 'command', 'config', 'options'}:
        raise ValueError('Invalid processing job fields')
    if type(data['schema']) is not int or data['schema'] != SCHEMA:
        raise ValueError('Unsupported processing job schema')
    if not isinstance(data['job_id'], str) or not re.fullmatch('[0-9a-f]{32}', data['job_id']):
        raise ValueError('Invalid job identity')
    if not isinstance(data['command'], str) or data['command'] not in OPTIONS or not isinstance(data['options'], dict):
        raise ValueError('Unsupported processing command/options')
    if not isinstance(data['config'], str) or not Path(data['config']).is_absolute():
        raise ValueError('Processing configuration must have an absolute path')
    for name, value in data['options'].items():
        kind = OPTIONS[data['command']].get(name)
        if kind is None or type(value) is not kind or (kind is int and value <= 0):
            raise ValueError(f'Invalid processing option: {name}')
    return data


def new_job(command, config, options=None):
    return validate_job(dict(schema=SCHEMA, job_id=uuid.uuid4().hex, command=command,
                             config=str(Path(config).resolve()), options=dict(options or {})))


def write_job(path, data):
    validate_job(data)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def result_path(path, job):
    return Path(path).parent / f"{job['job_id']}.result.json"


def validate_result(value, job):
    if (not isinstance(value, dict) or type(value.get('schema')) is not int or value.get('schema') != SCHEMA
            or value.get('job_id') != job['job_id'] or value.get('command') != job['command']
            or type(value.get('exit_code')) is not int or type(value.get('ok')) is not bool
            or value['ok'] != (value['exit_code'] == 0)):
        raise ValueError('Missing or mismatched processing result')
    return value


class EventStream:
    encoding = 'utf-8'

    def __init__(self, emit):
        self.emit, self.buffer = emit, ''

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer or '\r' in self.buffer:
            self.flush()
        return len(text)

    def flush(self):
        if self.buffer:
            text, self.buffer = self.buffer, ''
            self.emit('log', text=text)
            matches = re.findall(r'(\d+(?:\.\d+)?)%', text)
            if matches:
                self.emit('progress', fraction=min(1., max(0., float(matches[-1])/100)))

    def isatty(self):
        return False


def run(path, dispatch=None, sink=None):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Unsafe processing job file')
    job = validate_job(json.loads(path.read_text(encoding='utf-8')))
    destination = result_path(path, job)
    if destination.exists() or destination.is_symlink():
        raise ValueError('This job already has a result; queue a new job instead')
    lock = path.parent / f"{job['job_id']}.running"
    # A repeated descriptor cannot launch the same media job concurrently.
    # A forced termination leaves this small marker; new jobs get new identities.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    output = sys.stdout
    sequence = 0

    def emit(event, **fields):
        nonlocal sequence
        value = dict(schema=SCHEMA, job_id=job['job_id'], sequence=sequence, event=event, **fields)
        sequence += 1
        if sink:
            sink(value)
        else:
            output.write(json.dumps(value, ensure_ascii=True) + '\n')
            output.flush()

    emit('started', command=job['command'])
    stream = EventStream(emit)
    error = None
    code = 1
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            if dispatch is None:
                from .cli import main as dispatch
            arguments = ['--config', job['config'], job['command']]
            arguments += [f'--{name.replace("_", "-")}={value}' for name, value in job['options'].items()]
            code = int(dispatch(arguments) or 0)
    except KeyboardInterrupt:
        code, error = 130, 'Processing interrupted'
    except SystemExit as exc:
        code = exc.code if type(exc.code) is int else 1
        error = None if code == 0 else str(exc)
    except Exception as exc:
        code, error = 1, str(exc)
    finally:
        stream.flush()
        result = dict(schema=SCHEMA, job_id=job['job_id'], command=job['command'],
                      ok=code == 0, exit_code=code, error=error)
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.worker-result-', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(result, handle)
                handle.flush()
                os.fsync(handle.fileno())
            # Exclusive publication preserves preexisting/unrelated result files.
            # Filesystems without hard links use exclusive writing; a truncated
            # result can only fail validation and never unlock as success.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise
            except OSError:
                with destination.open('x', encoding='utf-8') as handle:
                    json.dump(result, handle)
            finally:
                temporary.unlink(missing_ok=True)
            if error:
                emit('error', message=error)
            emit('finished', ok=result['ok'], exit_code=code)
        finally:
            lock.unlink(missing_ok=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args(argv)
    try:
        return run(args.job)
    except (ValueError, OSError) as exc:
        print(f'Cannot start processing job: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
