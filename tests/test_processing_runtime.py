"""Download/extraction integrity and hardware choice without real engines/network."""
import copy
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from scout import __version__
from scout.processing_runtime import RuntimeStore, ReadyEngine, select_engine, validate_catalogue


def fixture(engine='linux-x86_64-cpu', *, malicious=None):
    content=b'verified native worker fixture'
    name='Scout-Worker/Scout-Worker'
    digest=lambda data:hashlib.sha256(data).hexdigest()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        member=tarfile.TarInfo(name if malicious is None else malicious)
        member.size=len(content)
        archive.addfile(member,io.BytesIO(content))
    payload=buffer.getvalue();middle=len(payload)//2
    chunks=[payload[:middle],payload[middle:]]
    parts=[dict(name=f'Scout-Runtime-{__version__}-{engine}.tar.gz.part{i:03d}',
                size=len(data),sha256=digest(data)) for i,data in enumerate(chunks,1)]
    spec=dict(entrypoint=name,archive='tar.gz',uncompressed_size=len(content),
              members={name:dict(size=len(content),sha256=digest(content),mode=0o755)},parts=parts)
    return dict(schema=1,application_version=__version__,runtimes={engine:spec}),dict(zip([p['name'] for p in parts],chunks))


class ProcessingRuntimeTests(unittest.TestCase):
    def setUp(self):
        import scout.processing_runtime as module
        before=module._active
        self.addCleanup(setattr,module,'_active',before)

    def test_distribution_builder_and_client_share_the_exact_contract(self):
        import importlib.util
        script=Path(__file__).resolve().parents[1]/'scripts/build-runtime.py'
        spec=importlib.util.spec_from_file_location('runtime_builder_integration',script)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle=root/'Scout-Worker';bundle.mkdir()
            (bundle/'Scout-Worker').write_bytes(b'fixture');(bundle/'Scout-Worker').chmod(0o755)
            output=root/'parts'
            catalog=builder.pack_runtime(bundle,output,'linux-x86_64-cpu',__version__,part_size=64)
            ready=RuntimeStore(catalog,root/'cache',offline_parts=output).ensure('linux-x86_64-cpu',probe=False)
            self.assertEqual(ready.executable.read_bytes(),b'fixture')

    def test_parts_are_verified_extracted_and_cached_before_any_execution(self):
        document,parts=fixture();calls=[]
        def fetch(url):
            self.assertTrue(url.startswith(f'https://github.com/illiw99/Scout-Studio/releases/download/v{__version__}/'))
            calls.append(url);return io.BytesIO(parts[url.rsplit('/',1)[-1]])
        with tempfile.TemporaryDirectory() as d:
            store=RuntimeStore(document,Path(d)/'cache',open_url=fetch)
            ready=store.ensure('linux-x86_64-cpu',probe=False)
            self.assertEqual(ready.executable.read_bytes(),b'verified native worker fixture')
            self.assertEqual(len(calls),2)
            store.ensure('linux-x86_64-cpu',probe=False);self.assertEqual(len(calls),2)
            ready.executable.write_bytes(b'bad')
            with patch.object(store,'probe',side_effect=AssertionError('damaged engine must not execute')):
                with self.assertRaisesRegex(ValueError,'damaged'):store.ensure('linux-x86_64-cpu')

    def test_corruption_and_cancellation_never_publish_an_engine(self):
        document,parts=fixture()
        for corrupt in (True,False):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'cache';cancel=Event()
                store=RuntimeStore(document,root,open_url=lambda url:io.BytesIO(b'bad' if corrupt else parts[url.rsplit('/',1)[-1]]))
                with self.assertRaises((ValueError,RuntimeError)):
                    store.ensure('linux-x86_64-cpu',probe=False,cancelled=cancel,
                                 progress=lambda *_:cancel.set() if not corrupt else None)
                self.assertFalse(store.folder('linux-x86_64-cpu').exists())
                self.assertEqual(list(root.glob('.prepare-*')),[])

    def test_repair_removes_only_a_damaged_downloaded_engine(self):
        document,parts=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);outside=root/'review.mlt';outside.write_bytes(b'user review')
            store=RuntimeStore(document,root/'cache',open_url=lambda url:io.BytesIO(parts[url.rsplit('/',1)[-1]]))
            ready=store.ensure('linux-x86_64-cpu',probe=False)
            ready.executable.write_bytes(b'damaged')
            store.repair_damaged()
            self.assertFalse(store.folder('linux-x86_64-cpu').exists())
            ready=store.ensure('linux-x86_64-cpu',probe=False)
            self.assertEqual(ready.executable.read_bytes(),b'verified native worker fixture')
            self.assertEqual(outside.read_bytes(),b'user review')

    def test_bootstrap_cli_delegates_before_loading_models_and_preserves_status(self):
        from scout import cli
        from types import SimpleNamespace
        ready=ReadyEngine('linux-x86_64-cpu',Path('/verified/worker'),('cpu',))
        for args in (['detect'],['model-pack','activate','--pack','approved@1.0.0']):
            with self.subTest(args=args),patch('scout.runtime.profile',return_value='bootstrap'), \
                 patch('scout.processing_runtime.RuntimeStore') as factory, \
                 patch('scout.cli._load',side_effect=AssertionError('GUI must not load a detector')), \
                 patch('subprocess.call',return_value=17) as call:
                factory.return_value.prepare.return_value=ready
                self.assertEqual(cli.main(args),17)
                self.assertEqual(call.call_args.args[0],[str(ready.executable),'--scout-cli',*args])
                self.assertEqual(call.call_args.kwargs['env']['PyInstaller_RESET_ENVIRONMENT'],'1')

    def test_lightweight_cli_model_list_does_not_prepare_an_engine(self):
        from scout import cli
        with tempfile.TemporaryDirectory() as d,patch('scout.runtime.profile',return_value='bootstrap'), \
             patch('scout.processing_runtime.RuntimeStore',side_effect=AssertionError('no download')):
            self.assertEqual(cli.main(['model-pack','list','--store',d]),0)

    def test_unsafe_archive_and_catalogue_paths_are_rejected(self):
        document,parts=fixture(malicious='../escape')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=RuntimeStore(document,root/'cache',open_url=lambda url:io.BytesIO(parts[url.rsplit('/',1)[-1]]))
            with self.assertRaisesRegex(ValueError,'Unsafe'):store.ensure('linux-x86_64-cpu',probe=False)
            self.assertFalse((root/'escape').exists())
        for path in ('../escape','/escape','C:/escape','a\\b','a/./b'):
            changed=copy.deepcopy(document);spec=changed['runtimes']['linux-x86_64-cpu']
            spec['members'][path]=spec['members'].pop(spec['entrypoint']);spec['entrypoint']=path
            with self.subTest(path=path),self.assertRaises(ValueError):validate_catalogue(changed)

    def test_hardware_selection_and_cpu_fallback(self):
        cpu,_=fixture();cuda,_=fixture('linux-x86_64-cuda')
        cpu['runtimes'].update(cuda['runtimes'])
        with patch('scout.processing_runtime.shutil.which',return_value='/usr/bin/nvidia-smi'):
            self.assertEqual(select_engine(cpu,target='linux-x86_64',
                run=lambda *a,**k:subprocess.CompletedProcess(a,0,'580.0\n','')),'linux-x86_64-cuda')
            self.assertEqual(select_engine(cpu,target='linux-x86_64',
                run=lambda *a,**k:subprocess.CompletedProcess(a,1,'','no driver')),'linux-x86_64-cpu')
        with tempfile.TemporaryDirectory() as d:
            store=RuntimeStore(cpu,Path(d))
            with patch('scout.processing_runtime.select_engine',return_value='linux-x86_64-cuda'), \
                 patch.object(store,'ensure',side_effect=[ReadyEngine('linux-x86_64-cuda',Path('cuda'),('cpu',)),
                                                         ReadyEngine('linux-x86_64-cpu',Path('cpu'),('cpu',))]) as ensure:
                ready=store.prepare();self.assertEqual(ready.engine_id,'linux-x86_64-cpu')
                self.assertEqual([c.args[0] for c in ensure.call_args_list],['linux-x86_64-cuda','linux-x86_64-cpu'])
        mac,_=fixture('macos-arm64')
        self.assertEqual(select_engine(mac,target='macos-arm64'),'macos-arm64')

    @unittest.skipIf(__import__('sys').platform=='win32','symlink privileges')
    def test_cached_symlinks_are_rejected_without_execution(self):
        document,parts=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=RuntimeStore(document,root/'cache',open_url=lambda url:io.BytesIO(parts[url.rsplit('/',1)[-1]]))
            ready=store.ensure('linux-x86_64-cpu',probe=False)
            outside=root/'outside';ready.executable.rename(outside);ready.executable.symlink_to(outside)
            with patch.object(store,'probe',side_effect=AssertionError('must not execute')):
                with self.assertRaisesRegex(ValueError,'links'):store.ensure('linux-x86_64-cpu')
