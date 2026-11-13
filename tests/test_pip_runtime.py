"""Private upstream setup is pinned, cancellable and isolated from user Python."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
import zipfile

from scout import __version__
from scout.pip_runtime import PipRuntimeStore,validate_catalog,PYTHON_VERSION
from scout.processing_runtime import RuntimeStore,platform_target


def fixture(root):
    assets=root/'assets';assets.mkdir()
    target=platform_target();engines=[target] if target.startswith('macos') else [target+'-cpu',target+'-cuda']
    entry='uv-0.9.7.data/scripts/'+('uv.exe' if os.name=='nt' else 'uv')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:archive.writestr(entry,b'trusted fixture installer')
    payload=buffer.getvalue();sha=lambda data:hashlib.sha256(data).hexdigest()
    contents={'worker_launcher.py':b'# fixture launcher',f'scout_flag-{__version__}-py3-none-any.whl':b'fixture wheel'}
    runtimes={}
    for engine in engines:
        name=f'requirements-{engine}.txt';contents[name]=b'fixture==1 --hash=sha256:'+b'a'*64
        runtimes[engine]=dict(requirements=name,wheel=f'scout_flag-{__version__}-py3-none-any.whl',entrypoint='worker_launcher.py')
    for name,data in contents.items():(assets/name).write_bytes(data)
    catalogue=dict(schema=2,application_version=__version__,python=PYTHON_VERSION,target=target,
        bootstrap=dict(url='https://files.pythonhosted.org/packages/uv-0.9.7-fixture.whl',size=len(payload),sha256=sha(payload),member=entry),
        files={name:dict(size=len(data),sha256=sha(data)) for name,data in contents.items()},runtimes=runtimes)
    return catalogue,assets,payload,engines[0]


class PipRuntimeTests(unittest.TestCase):
    def setup_store(self,root):
        catalogue,assets,payload,engine=fixture(root)
        store=RuntimeStore(catalogue,root/'private',offline_parts=assets,open_url=lambda url:io.BytesIO(payload))
        commands=[]
        def run(command,**kwargs):
            commands.append(list(map(str,command)))
            base=store.python_root/'fixture-python';base.mkdir(parents=True,exist_ok=True)
            (base/'stdlib.py').write_text('# trusted stdlib')
            if 'venv' in command:
                env=Path(command[-1]);python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                python.parent.mkdir(parents=True,exist_ok=True);python.write_bytes(b'private interpreter')
            elif '-c' in command:(store.root/'setup.log').write_text(str(base.resolve())+'\n')
        spec=catalogue['runtimes'][engine]
        report=dict(schema=1,ok=True,devices=['cpu'],nms=True,inference=True,engine_id=engine,
            application_version=__version__,requirements_sha256=catalogue['files'][spec['requirements']]['sha256'],
            wheel_sha256=catalogue['files'][spec['wheel']]['sha256'])
        store._run=run;store.probe=lambda *args,**kwargs:report
        return store,engine,commands

    def test_only_private_managed_python_and_hash_pinned_binary_installs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sentinel=root/'system-python';sentinel.write_text('untouched')
            store,engine,commands=self.setup_store(root)
            self.assertIsInstance(store,PipRuntimeStore)
            ready=store.ensure(engine)
            self.assertEqual(ready.command[1:3],['-I','-B'])
            self.assertIn('--managed-python',commands[0]);self.assertIn(PYTHON_VERSION,commands[0])
            for command in commands[1:3]:
                self.assertIn('--require-hashes',command);self.assertIn('--no-deps',command)
                self.assertIn('--only-binary',command);self.assertIn('--python',command)
                self.assertEqual(command[command.index('--python')+1],str(ready.executable))
            self.assertEqual(sentinel.read_text(),'untouched')
            before=len(commands);store.ensure(engine);self.assertEqual(len(commands),before)

    def test_changed_installed_package_is_rejected_and_repaired_without_user_data_loss(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store,engine,_=self.setup_store(root);ready=store.ensure(engine)
            models=root/'user-models';models.mkdir();(models/'keep.json').write_text('keep')
            ready.executable.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):store.ensure(engine)
            store.repair_damaged();self.assertFalse(store.folder(engine).exists())
            self.assertEqual((models/'keep.json').read_text(),'keep')
            self.assertTrue(store.ensure(engine).executable.is_file())

    def test_cancelled_package_install_never_publishes_ready_environment(self):
        with tempfile.TemporaryDirectory() as d:
            store,engine,_=self.setup_store(Path(d));original=store._run
            def fail(command,**kwargs):
                if 'install' in command:raise RuntimeError('cancelled')
                original(command,**kwargs)
            store._run=fail
            with self.assertRaisesRegex(RuntimeError,'cancelled'):store.ensure(engine)
            self.assertFalse(store.folder(engine).exists())
            cancelled=Event();cancelled.set()
            with self.assertRaisesRegex(RuntimeError,'cancelled'):store.ensure(engine,cancelled=cancelled)

    def test_recipe_and_upstream_corruption_cannot_execute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store,engine,commands=self.setup_store(root)
            name=store.catalogue['runtimes'][engine]['requirements'];(store.assets/name).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):store.ensure(engine)
            self.assertFalse(commands)
        with tempfile.TemporaryDirectory() as d:
            store,engine,commands=self.setup_store(Path(d));store.open_url=lambda url:io.BytesIO(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):store.ensure(engine)
            self.assertFalse(commands)

    def test_environment_ignores_user_pip_uv_and_python_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            store,_,_=self.setup_store(Path(d))
            with patch.dict(os.environ,PIP_TARGET='/outside',UV_PYTHON='/system/python',PYTHONPATH='/evil'):
                env=store._environment()
            self.assertNotIn('PIP_TARGET',env);self.assertNotIn('UV_PYTHON',env);self.assertNotIn('PYTHONPATH',env)
            self.assertEqual(env['UV_PYTHON_INSTALL_DIR'],str(store.python_root))

    def test_external_links_and_probe_identity_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store,engine,_=self.setup_store(root);store.ensure(engine)
            outside=root/'outside';outside.write_text('not part of environment')
            link=store.folder(engine)/'external'
            try:link.symlink_to(outside)
            except OSError:self.skipTest('Symlink capability unavailable')
            with self.assertRaisesRegex(ValueError,'escapes'):store.ensure(engine)
            with patch.object(RuntimeStore,'probe',return_value={'devices':['cpu']}):
                with self.assertRaisesRegex(ValueError,'identity'):PipRuntimeStore.probe(store,Path('python'),engine)

    def test_shared_python_changes_cannot_be_blessed_by_second_engine(self):
        with tempfile.TemporaryDirectory() as d:
            store,engine,commands=self.setup_store(Path(d));store.ensure(engine)
            others=set(store.catalogue['runtimes'])-{engine}
            if not others:self.skipTest('Single-engine platform')
            (store.python_root/'fixture-python'/'changed.py').write_text('changed')
            before=len(commands)
            with self.assertRaisesRegex(ValueError,'changed'):store.ensure(others.pop())
            self.assertEqual(len(commands),before)
            store.repair_damaged();self.assertFalse(store.python_root.exists())

    def test_malformed_receipt_and_permission_damage_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            store,engine,_=self.setup_store(Path(d));ready=store.ensure(engine)
            (store.folder(engine)/'ready.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'receipt'):store.ensure(engine)
            store.repair_damaged();ready=store.ensure(engine)
            if os.name=='posix':
                ready.executable.chmod(0o700)
                with self.assertRaisesRegex(ValueError,'changed'):store.ensure(engine)
                store.repair_damaged();self.assertFalse(store.folder(engine).exists())

    def test_catalogue_rejects_unpinned_targets_paths_and_bootstrap_hosts(self):
        with tempfile.TemporaryDirectory() as d:
            catalogue,_,_,_=fixture(Path(d))
            for field,value in [('python','3.12'),('target','other')]:
                data=copy.deepcopy(catalogue);data[field]=value
                with self.assertRaises(ValueError):validate_catalog(data)
            data=copy.deepcopy(catalogue);data['bootstrap']['url']='https://example.org/uv-0.9.7.whl'
            with self.assertRaises(ValueError):validate_catalog(data)
            data=copy.deepcopy(catalogue);data['files']['../outside']={'size':1,'sha256':'a'*64}
            with self.assertRaises(ValueError):validate_catalog(data)


if __name__=='__main__':unittest.main()
