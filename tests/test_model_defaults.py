"""Offline default bootstrapping preserves explicit model selections and data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scout.model_packs import ModelStore, sha
from scout.model_defaults import prepare_bundle, ensure_default


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'CPU graph fixture needs inference extra')
class BundledDefaultTests(unittest.TestCase):
    def fixture(self, root, version='1.0.0'):
        from test_model_packs import fixture
        archive, digest, _, _ = fixture(root, version, modern=True, context=True)
        bundle = prepare_bundle(archive, digest, root / ('bundle-' + version))
        return bundle

    def test_first_use_installs_default_and_repeated_use_does_not_reimport(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=self.fixture(root);store=ModelStore(root/'store')
            self.assertEqual(set(p.name for p in bundle.iterdir()), {'receipt.json','model-pack.zip'})
            receipt=json.loads((bundle/'receipt.json').read_text())
            self.assertEqual(sha(bundle/'model-pack.zip'),receipt['sha256'])
            self.assertEqual(ensure_default(store,bundle),'synthetic@1.0.0')
            pointer=(store.root/'active.json').read_bytes()
            with patch.object(store,'install',side_effect=AssertionError('unexpected reimport')):
                self.assertEqual(ensure_default(store,bundle),'synthetic@1.0.0')
            self.assertEqual((store.root/'active.json').read_bytes(),pointer)

    def test_managed_upgrade_preserves_rollback_and_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store');first=self.fixture(root)
            ensure_default(store,first);second=self.fixture(root,'1.1.0')
            self.assertEqual(ensure_default(store,second),'synthetic@1.1.0')
            self.assertEqual(store.rollback(),'synthetic@1.0.0')
            self.assertEqual(ensure_default(store,second),'synthetic@1.0.0')
            self.assertEqual(store.active(),'synthetic@1.0.0')

    def test_unmanaged_active_model_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);first=self.fixture(root);second=self.fixture(root,'1.1.0')
            store=ModelStore(root/'store')
            store.install(first/'model-pack.zip');store.activate('synthetic@1.0.0')
            before=(store.root/'active.json').read_bytes()
            self.assertEqual(ensure_default(store,second),'synthetic@1.0.0')
            self.assertEqual((store.root/'active.json').read_bytes(),before)

    def test_damage_cannot_change_active_selection_or_overwrite_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=self.fixture(root);store=ModelStore(root/'store')
            ensure_default(store,bundle);pointer=(store.root/'active.json').read_bytes()
            (bundle/'model-pack.zip').write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError,'checksum'):ensure_default(store,bundle)
            self.assertEqual((store.root/'active.json').read_bytes(),pointer)
            with self.assertRaisesRegex(ValueError,'new baseline'):prepare_bundle(bundle/'model-pack.zip','a'*64,bundle)

    def test_raced_explicit_activation_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=self.fixture(root);second=self.fixture(root,'1.1.0');store=ModelStore(root/'store')
            store.install(second/'model-pack.zip')
            activate=store.activate
            def race(selection,**kwargs):
                activate('synthetic@1.1.0')
                return activate(selection,**kwargs)
            with patch.object(store,'activate',side_effect=race):
                self.assertEqual(ensure_default(store,bundle),'synthetic@1.1.0')
            self.assertEqual(store.active(),'synthetic@1.1.0')

    def test_choice_changed_during_active_read_is_not_adopted_as_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store')
            first=self.fixture(root);ensure_default(store,first)
            upgrade=self.fixture(root,'1.1.0');manual=self.fixture(root,'1.2.0')
            store.install(manual/'model-pack.zip');read=store.active;activate=store.activate
            calls=[]
            def race():
                value=read()
                if not calls:
                    calls.append(True);activate('synthetic@1.2.0')
                return value
            with patch.object(store,'active',side_effect=race):
                self.assertEqual(ensure_default(store,upgrade),'synthetic@1.2.0')
            self.assertEqual(read(),'synthetic@1.2.0')
