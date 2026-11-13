"""Private, pinned upstream Python/package installation for the native app.

The GUI orchestrates setup off-thread. Only the separate Python worker imports
processing dependencies. No system interpreter or package installation is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import ssl
import stat
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPSHandler
import zipfile

from . import __version__
from .processes import background_options
from .storage import data_lock

PYTHON_VERSION = '3.12.12'
MAX_FILE = 32 * 1024 * 1024


def validate_catalog(document, *, version=None):
    from .processing_runtime import ENGINE_IDS, _hash, _relative
    if (not isinstance(document,dict) or type(document.get('schema')) is not int or document['schema']!=2
            or document.get('application_version') != (version or __version__)
            or document.get('python') != PYTHON_VERSION
            or document.get('target') not in ('linux-x86_64','windows-x86_64','macos-arm64','macos-x86_64')):
        raise ValueError('Incompatible pip processing catalogue')
    files=document.get('files');runtimes=document.get('runtimes');bootstrap=document.get('bootstrap')
    target=document['target']
    expected={target} if target.startswith('macos-') else {target+'-cpu',target+'-cuda'}
    if (not isinstance(files,dict) or not 3<=len(files)<=8 or not isinstance(runtimes,dict)
            or set(runtimes)!=expected or not set(runtimes)<=ENGINE_IDS):
        raise ValueError('Invalid pip processing assets/targets')
    for name,record in files.items():
        if (len(_relative(name).parts)!=1 or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',name) or not isinstance(record,dict)
                or set(record)!={'size','sha256'} or type(record['size']) is not int
                or not 0<record['size']<=MAX_FILE or not _hash(record['sha256'])):
            raise ValueError('Invalid pip processing asset identity')
    used=set()
    for engine_id,recipe in runtimes.items():
        if (not isinstance(recipe,dict) or set(recipe)!={'requirements','wheel','entrypoint'}
                or any(name not in files for name in recipe.values())
                or recipe['requirements']!=f'requirements-{engine_id}.txt'
                or recipe['wheel']!=f'scout_flag-{document["application_version"]}-py3-none-any.whl'
                or recipe['entrypoint']!='worker_launcher.py'):
            raise ValueError('Invalid pip processing recipe')
        used.update(recipe.values())
    if used!=set(files):raise ValueError('Unreferenced pip processing assets')
    if (not isinstance(bootstrap,dict) or set(bootstrap)!={'url','size','sha256','member'}
            or not isinstance(bootstrap['url'],str)):
        raise ValueError('Invalid Python installer bootstrap')
    url=urlsplit(bootstrap['url'])
    if (url.scheme!='https' or url.hostname!='files.pythonhosted.org' or url.port not in (None,443)
            or url.username or url.password or url.query or url.fragment
            or not Path(url.path).name.startswith('uv-0.9.7-') or not url.path.endswith('.whl')
            or type(bootstrap['size']) is not int or not 0<bootstrap['size']<=MAX_FILE
            or not _hash(bootstrap['sha256'])
            or _relative(bootstrap['member']).name!=('uv.exe' if target.startswith('windows') else 'uv')):
        raise ValueError('Unsupported upstream Python installer')
    return document


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        parsed=urlsplit(newurl)
        if parsed.scheme!='https' or parsed.hostname!='files.pythonhosted.org':
            raise ValueError('Unsupported upstream bootstrap redirect')
        return super().redirect_request(request,fp,code,message,headers,newurl)


def _open(url):
    import certifi
    context=ssl.create_default_context(cafile=certifi.where())
    return build_opener(_Redirect(),HTTPSHandler(context=context)).open(
        Request(url,headers={'User-Agent':'Scout-Studio-setup'}),timeout=30)


class PipRuntimeStore:
    def __init__(self,catalogue,root=None,*,offline_parts=None,open_url=None):
        from .processing_runtime import cache_root,catalogue_path
        self.catalogue=validate_catalog(catalogue)
        self.root=Path(root or cache_root()).expanduser().absolute()
        self.python_root=self.root/f'python-{__version__}'
        path=catalogue_path()
        self.assets=Path(offline_parts).resolve() if offline_parts is not None else path.resolve().parent if path else None
        self.open_url=open_url or _open

    @staticmethod
    def _cancel(cancelled):
        from .processing_runtime import RuntimeStore
        RuntimeStore._cancel(cancelled)

    def folder(self,engine_id):
        recipe=self.catalogue['runtimes'][engine_id]
        identity=dict(recipe=recipe,files={name:self.catalogue['files'][name] for name in recipe.values()},
            bootstrap=self.catalogue['bootstrap'],python=self.catalogue['python'])
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return self.root/f'pip-{__version__}-{engine_id}-{digest[:16]}'

    def _asset(self,name):
        from .processing_runtime import _digest,_linked
        if self.assets is None:raise ValueError('Missing bundled pip recipe assets')
        path=self.assets/name
        # Frozen macOS resources may be framework links into Resources.
        if _linked(path):
            import sys
            if not getattr(sys,'frozen',False):raise ValueError('Pip recipe asset must not be a link')
            path=path.resolve()
        record=self.catalogue['files'][name]
        if not path.is_file() or path.stat().st_size!=record['size'] or _digest(path)!=record['sha256']:
            raise ValueError('Bundled pip recipe asset changed')
        return path

    def _environment(self):
        from .processing_runtime import worker_environment
        env={key:value for key,value in worker_environment().items()
             if not key.startswith(('PIP_','UV_')) and key not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')}
        env.update(UV_PYTHON_INSTALL_DIR=str(self.python_root),
            UV_PYTHON_DOWNLOADS='automatic',UV_CACHE_DIR=str(self.root/'package-cache'),UV_LINK_MODE='copy',
            UV_COMPILE_BYTECODE='0',PIP_CONFIG_FILE=os.devnull)
        return env

    def _run(self,command,*,cancelled=None,label='setup',timeout=2400):
        from .processing_runtime import _linked
        log=self.root/'setup.log'
        if _linked(log):raise ValueError('Unsafe setup log')
        with log.open('wb') as output:
            process=subprocess.Popen([str(arg) for arg in command],stdout=output,stderr=subprocess.STDOUT,
                env=self._environment(),start_new_session=os.name=='posix',**background_options())
            begun=time.monotonic()
            try:
                while process.poll() is None:
                    self._cancel(cancelled)
                    if time.monotonic()-begun>timeout:raise RuntimeError(f'{label} timed out')
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                    elif os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,**background_options())
                    else:process.kill()
                    process.wait(timeout=10)
        if process.returncode:raise RuntimeError(f'{label} failed: '+log.read_text(errors='replace')[-2000:])

    def _bootstrap(self,cancelled=None):
        from .processing_runtime import _digest,_linked
        spec=self.catalogue['bootstrap'];folder=self.root/('bootstrap-'+spec['sha256'][:16])
        if _linked(folder):raise ValueError('Unsafe Python bootstrap directory')
        executable=folder/('uv.exe' if os.name=='nt' else 'uv')
        receipt=folder/'identity.json'
        if folder.exists():
            wheel=folder/'upstream.whl'
            if (not receipt.is_file() or receipt.stat().st_size>1024 or _linked(receipt) or _linked(executable)
                    or not wheel.is_file() or _linked(wheel) or _digest(wheel)!=spec['sha256']):
                raise ValueError('Python bootstrap cache changed; repair setup')
            expected=hashlib.sha256()
            with zipfile.ZipFile(wheel) as archive,archive.open(spec['member']) as binary:
                for block in iter(lambda:binary.read(1024*1024),b''):expected.update(block)
            if _digest(executable)!=expected.hexdigest():raise ValueError('Python bootstrap executable changed; repair setup')
            return executable
        with tempfile.TemporaryDirectory(prefix='.python-bootstrap-',dir=self.root) as temporary:
            stage=Path(temporary);wheel=stage/'upstream.whl';size=0;digest=hashlib.sha256()
            with self.open_url(spec['url']) as source,wheel.open('xb') as output:
                while True:
                    self._cancel(cancelled);block=source.read(1024*1024)
                    if not block:break
                    size+=len(block)
                    if size>spec['size']:raise ValueError('Oversized upstream Python installer')
                    digest.update(block);output.write(block)
            if size!=spec['size'] or digest.hexdigest()!=spec['sha256']:raise ValueError('Python installer checksum mismatch')
            with zipfile.ZipFile(wheel) as archive:
                entries=[entry for entry in archive.infolist() if entry.filename==spec['member']]
                if (len(entries)!=1 or not 0<entries[0].file_size<=128*1024*1024
                        or stat.S_ISLNK(entries[0].external_attr>>16)):
                    raise ValueError('Invalid upstream Python installer wheel')
                binary=stage/executable.name
                with archive.open(entries[0]) as source,binary.open('xb') as output:shutil.copyfileobj(source,output)
            binary.chmod(0o700)
            # Keep upstream metadata/licenses inside the verified original wheel.
            (stage/'identity.json').write_text(json.dumps(dict(executable_sha256=_digest(binary),upstream_sha256=spec['sha256'])))
            stage.rename(folder)
        return executable

    def _inventory(self,root,cancelled=None):
        from .processing_runtime import _digest,_linked
        actual={};total=0;allowed=root.resolve()
        for path in root.rglob('*'):
            self._cancel(cancelled)
            name=path.relative_to(root).as_posix()
            if root==self._inventory_root and name=='ready.json':continue
            if _linked(path):
                resolved=path.resolve(strict=True)
                in_base=(root==self._inventory_root and resolved.is_relative_to(self._allowed_base.resolve()))
                if not resolved.is_relative_to(allowed) and not in_base:
                    raise ValueError('Python environment link escapes verified setup')
                actual[name]=dict(link=os.readlink(path))
            elif path.is_file():
                total+=path.stat().st_size
                if total>24*1024**3:raise ValueError('Installed environment exceeds size bound')
                actual[name]=dict(size=path.stat().st_size,sha256=_digest(path))
                if os.name=='posix':actual[name]['mode']=stat.S_IMODE(path.stat().st_mode)
            if len(actual)>100000:raise ValueError('Too many installed environment files')
        return actual

    def _verify(self,folder,spec,cancelled=None):
        from .processing_runtime import _linked,_relative
        self._cancel(cancelled)
        receipt=folder/'ready.json'
        if _linked(folder) or _linked(receipt) or not receipt.is_file() or receipt.stat().st_size>32*1024*1024:
            raise ValueError('Incomplete private processing environment; repair setup')
        value=json.loads(receipt.read_text())
        if not isinstance(value,dict):raise ValueError('Invalid processing receipt; repair setup')
        if value.get('catalogue_sha256')!=self._catalogue_hash():raise ValueError('Processing recipe identity changed')
        identity_file=folder/'engine-id.json'
        if _linked(identity_file) or not identity_file.is_file() or identity_file.stat().st_size>1024:
            raise ValueError('Invalid installed processing identity')
        engine_id=next((engine for engine in self.catalogue['runtimes'] if self.folder(engine)==folder),None)
        expected=dict(engine_id=engine_id,requirements_sha256=self.catalogue['files'][spec['requirements']]['sha256'],
            wheel_sha256=self.catalogue['files'][spec['wheel']]['sha256'])
        if json.loads(identity_file.read_text())!=expected:raise ValueError('Installed processing identity differs')
        base=self.root/_relative(value.get('python_base'))
        if (_linked(self.python_root) or _linked(base) or not base.is_dir()
                or not base.resolve().is_relative_to(self.python_root.resolve())):
            raise ValueError('Invalid private Python location')
        self._inventory_root=folder
        self._allowed_base=base
        if self._inventory(folder,cancelled)!=value.get('files') or self._inventory(base,cancelled)!=value.get('python_files'):
            raise ValueError('Private processing environment changed; repair setup')
        return value

    def _catalogue_hash(self):
        return hashlib.sha256(json.dumps(self.catalogue,sort_keys=True).encode()).hexdigest()

    def worker_command(self,executable,engine_id):
        return [str(executable),'-I','-B',str(self.folder(engine_id)/'worker_launcher.py')]

    def probe(self,executable,engine_id,cancelled=None):
        from .processing_runtime import RuntimeStore
        value=RuntimeStore.probe(self,executable,engine_id,cancelled)
        recipe=self.catalogue['runtimes'][engine_id]
        for key,asset in [('requirements_sha256','requirements'),('wheel_sha256','wheel')]:
            if value.get(key)!=self.catalogue['files'][recipe[asset]]['sha256']:
                raise ValueError('Processing probe recipe identity differs')
        return value

    def ensure(self,engine_id,*,cancelled=None,progress=None,probe=True):
        from .processing_runtime import ReadyEngine,_linked
        notify=progress or (lambda message,percent:None)
        self._cancel(cancelled)
        if _linked(self.root):raise ValueError('Unsafe processing setup root')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if os.name=='posix':self.root.chmod(0o700)
        spec=self.catalogue['runtimes'][engine_id];folder=self.folder(engine_id)
        lock='.pip-setup.lock'
        if _linked(self.root/lock):raise ValueError('Unsafe processing setup lock')
        with data_lock(self.root,lock):
            python=folder/('venv/Scripts/python.exe' if self.catalogue['target'].startswith('windows') else 'venv/bin/python')
            if folder.exists():
                notify('Checking installed processing packages',0)
                self._verify(folder,spec,cancelled)
            else:
                for asset in spec.values():self._asset(asset)
                # A second engine must not bless an altered shared interpreter.
                # Interrupted first installs have no trusted receipt and need repair.
                if self.python_root.exists():
                    trusted=False
                    for existing_id,existing_spec in self.catalogue['runtimes'].items():
                        existing=self.folder(existing_id)
                        if existing.exists():
                            self._verify(existing,existing_spec,cancelled)
                            trusted=True
                    if not trusted:raise ValueError('Unverified private Python; repair setup')
                uv=self._bootstrap(cancelled)
                if _linked(self.python_root):raise ValueError('Unsafe private Python directory')
                folder.mkdir(mode=0o700)
                try:
                    notify('Installing private Python',10)
                    self._run([uv,'--no-config','--no-cache','venv','--managed-python','--python',PYTHON_VERSION,
                               folder/'venv'],cancelled=cancelled,label='Private Python setup')
                    notify('Downloading and installing processing packages',30)
                    # Match pip across the two official indexes; exact wheel
                    # hashes prevent dependency-confusion substitutions.
                    self._run([uv,'--no-config','--no-cache','pip','install','--python',python,'--require-hashes',
                        '--index-strategy','unsafe-best-match','--only-binary',':all:','--no-deps','-r',self._asset(spec['requirements'])],
                        cancelled=cancelled,label='Processing package installation')
                    application=folder/'application.txt'
                    application.write_text(self._asset(spec['wheel']).as_uri()+' --hash=sha256:'+self.catalogue['files'][spec['wheel']]['sha256']+'\n')
                    self._run([uv,'--no-config','--no-cache','pip','install','--python',python,'--require-hashes',
                        '--no-deps','--only-binary',':all:','-r',application],cancelled=cancelled,label='Scout processing code setup')
                    shutil.copyfile(self._asset(spec['entrypoint']),folder/'worker_launcher.py')
                    (folder/'engine-id.json').write_text(json.dumps(dict(engine_id=engine_id,
                        requirements_sha256=self.catalogue['files'][spec['requirements']]['sha256'],
                        wheel_sha256=self.catalogue['files'][spec['wheel']]['sha256'])))
                    # Query only the freshly installed private interpreter.
                    self._run([python,'-I','-B','-c','import sys; print(sys.base_prefix)'],cancelled=cancelled,label='Python location check')
                    base=Path((self.root/'setup.log').read_text().strip()).resolve()
                    if not base.is_relative_to(self.python_root.resolve()):raise ValueError('Setup used a nonprivate Python')
                    notify('Checking processing compatibility',85)
                    report=self.probe(python,engine_id,cancelled)
                    self._inventory_root=folder
                    self._allowed_base=base
                    receipt=dict(catalogue_sha256=self._catalogue_hash(),python_base=base.relative_to(self.root.resolve()).as_posix(),
                        files=self._inventory(folder,cancelled),python_files=self._inventory(base,cancelled),devices=report['devices'])
                    temporary=folder/'ready.json.tmp';temporary.write_text(json.dumps(receipt,sort_keys=True));temporary.replace(folder/'ready.json')
                except BaseException:
                    shutil.rmtree(folder,ignore_errors=True)
                    raise
            report=self.probe(python,engine_id,cancelled) if probe else None
            devices=tuple(report['devices']) if report else ('cpu',)
            notify('Processing is ready',100)
            return ReadyEngine(engine_id,python,devices,('-I','-B',str(folder/'worker_launcher.py')))

    def prepare(self,**kwargs):
        from .processing_runtime import RuntimeStore
        return RuntimeStore.prepare(self,**kwargs)

    def repair_damaged(self,*,cancelled=None,progress=None):
        from .processing_runtime import _linked
        if _linked(self.root):raise ValueError('Unsafe processing setup root')
        if not self.root.exists():return
        def remove(path):
            if path.is_symlink() or path.is_file():path.unlink()
            elif _linked(path):path.rmdir()
            elif path.exists():shutil.rmtree(path)
        if _linked(self.root/'.pip-setup.lock'):raise ValueError('Unsafe processing setup lock')
        with data_lock(self.root,'.pip-setup.lock'):
            damaged=False;found=False
            for engine_id,spec in self.catalogue['runtimes'].items():
                folder=self.folder(engine_id)
                if not folder.exists() and not _linked(folder):continue
                found=True
                self._cancel(cancelled)
                try:self._verify(folder,spec,cancelled)
                except (ValueError,OSError,KeyError):damaged=True
            # Retry also repairs a failed/interrupted Python bootstrap. These
            # are strictly this version's private packages, never user models.
            if damaged or not found:
                if progress:progress('Repairing private processing packages',0)
                for engine_id in self.catalogue['runtimes']:
                    self._cancel(cancelled);remove(self.folder(engine_id))
                remove(self.python_root)
            bootstrap=self.root/('bootstrap-'+self.catalogue['bootstrap']['sha256'][:16])
            if bootstrap.exists():
                try:self._bootstrap(cancelled)
                except (ValueError,OSError,KeyError,zipfile.BadZipFile):remove(bootstrap)
