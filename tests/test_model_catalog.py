"""Decision-only catalogue ships no pretrained weights and remains lazy."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from scout.model_catalog import read,prepare_default_bundle
from scout.model_packs import ModelStore,fallback_to_cpu
from scout.config import Config

ROOT=Path(__file__).resolve().parents[1]/'models'

class ModelCatalogTests(unittest.TestCase):
    def test_windows_git_checkout_preserves_catalogue_asset_bytes(self):
        import shutil,subprocess
        from scout.model_catalog import release_files
        repository=ROOT.parent
        if not shutil.which('git'):
            self.skipTest('Git checkout validation needs Git')
        check=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repository,
                             capture_output=True,text=True)
        if check.returncode or Path(check.stdout.strip()).resolve()!=repository.resolve():
            self.skipTest('Source archive has no repository index')
        files=[str(p.relative_to(repository)) for p in release_files(ROOT)]
        with tempfile.TemporaryDirectory() as d:
            # Use this repository's index and attributes without changing it.
            subprocess.run(['git','-c','core.autocrlf=true','checkout-index',
                            '--prefix='+Path(d).as_posix()+'/', '--',*files],
                           cwd=repository,check=True,capture_output=True)
            self.assertEqual(read(Path(d)/'models'),read(ROOT))

    def test_catalogue_bundles_only_trained_decision_assets_without_download(self):
        from scout.model_defaults import ensure_default
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=ModelStore(root/'store')
            with patch('scout.inference.reference_weights',side_effect=AssertionError('startup must not download')):
                bundle=prepare_default_bundle(ROOT,root/'bundle')
                with zipfile.ZipFile(bundle/'model-pack.zip') as archive:
                    self.assertEqual(set(archive.namelist()),{'manifest.json','classifier.json'})
                self.assertEqual(ensure_default(store,bundle),'scout-reviewed-precision@0.4.0')
            spec=json.loads((store.folder(store.active())/'manifest.json').read_text())
            self.assertEqual(spec['schema'],3)
            self.assertEqual(spec['backend'],'managed')

    def test_managed_pack_fetches_pinned_detector_only_when_applied_and_falls_back(self):
        from scout.model_defaults import ensure_default
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=ModelStore(root/'store');bundle=prepare_default_bundle(ROOT,root/'bundle');ensure_default(store,bundle)
            weights=root/'yolo.pt';weights.write_bytes(b'fixture')
            cfg=Config(device='auto')
            with patch('scout.inference.reference_weights',return_value=weights) as fetch, \
                 patch('scout.model_packs._cuda_status',return_value=(True,'')), \
                 patch('scout.inference.prepare_graph',return_value=root/'model.xml') as export, \
                 patch('scout.inference.version',return_value='present'):
                store.apply(cfg,store.active());self.assertEqual(cfg.device,'cuda:0');export.assert_not_called()
                self.assertEqual(fetch.call_args.args[1],'yolo26n.pt')
                self.assertTrue(cfg.allow_gpu_fallback)
                fallback_to_cpu(cfg)
                self.assertEqual(cfg.device,'cpu');self.assertEqual(cfg.detector_backend,'openvino')
                self.assertEqual(cfg.model,str(root/'model.xml'));export.assert_called_once()
            self.assertEqual(hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest(),read(ROOT)['files']['scout-reviewed-precision/0.4.0/classifier.json']['sha256'])

    def test_catalogue_path_and_checksum_rejection(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'models';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('play.pkl'))
            file=root/'scout-reviewed-precision/0.4.0/classifier.json';file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):read(root)
            document=json.loads((root/'catalog.json').read_text());document['files']['../outside']={'size':1,'sha256':'0'*64};(root/'catalog.json').write_text(json.dumps(document))
            with self.assertRaises(ValueError):read(root)

    @unittest.skipIf(__import__('sys').platform == 'win32', 'directory symlinks need privileges')
    def test_frozen_distribution_root_link_accepts_only_real_nested_assets(self):
        import shutil,sys
        from scout.model_catalog import directory
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);resources=root/'Resources/models'
            shutil.copytree(ROOT,resources)
            framework=root/'Frameworks';framework.mkdir()
            (framework/'models').symlink_to(resources,target_is_directory=True)
            with patch.object(sys,'_MEIPASS',str(framework),create=True):
                self.assertEqual(directory(),resources.resolve())
                self.assertEqual(read()['schema'],1)
                asset=resources/'person-transformer/1.0.0/head.onnx'
                outside=root/'head.onnx';asset.rename(outside);asset.symlink_to(outside)
                with self.assertRaisesRegex(ValueError,'links'):read()

    def test_installer_override_rejects_generic_graph_pack(self):
        import importlib.util
        if not (importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime')):
            self.skipTest('CPU graph fixture needs inference extra')
        from test_model_packs import fixture
        from scout.model_defaults import prepare_installer_bundle
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive,digest,_,_=fixture(root,'1.0.0',modern=True,context=True)
            with self.assertRaisesRegex(ValueError,'decision-only'):
                prepare_installer_bundle(archive,digest,root/'rejected')
            self.assertFalse((root/'rejected').exists())
            bundle=prepare_default_bundle(ROOT,root/'source')
            from scout.model_packs import sha
            accepted=prepare_installer_bundle(bundle/'model-pack.zip',sha(bundle/'model-pack.zip'),root/'accepted')
            self.assertTrue((accepted/'receipt.json').is_file())
