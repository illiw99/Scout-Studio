import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scout import cli, collect, storage
from scout.config import Config
from scout.groundtruth import Marked
from scout.media import MediaInfo


class GameStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def load(self, **values):
        args = dict(config=None, match=None, source_dir=None, out_dir=None, limit=None)
        args.update(values)
        return cli._load(argparse.Namespace(**args))

    def test_default_game_paths_and_explicit_overrides(self):
        cfg = Config(match='Game')
        game = self.root / 'media/Game'
        self.assertEqual(cfg.game_dir, game)
        self.assertEqual(cfg.path(cfg.out_dir), game / '.scout')
        self.assertEqual(cfg.path(cfg.cache_dir), game / '.scout/cache')
        self.assertEqual(cfg.path(cfg.fields_dir), game / '.scout/fields')
        self.assertEqual(cfg.path('models/play.pkl'), self.root / 'models/play.pkl')
        self.assertEqual(self.load(match='Game', out_dir='custom').path('custom'), self.root / 'custom')
        self.assertFalse((self.root / 'out').exists())
        self.assertFalse((game / '.scout').exists())

    def test_registry_external_folder_and_stale_saved_source(self):
        external = self.root / 'external/recording'
        external.mkdir(parents=True)
        storage.write_library(self.root, {'Game': str(external)})
        cfg = Config(source_dir=str(self.root / 'unrelated'), match='Old')
        config = self.root / 'scout.json'
        cfg.save(config)
        selected = self.load(config=str(config), match='Game')
        self.assertEqual(selected.source_dir, str(external))
        self.assertEqual(selected.path(selected.out_dir), external / '.scout')
        fallback = self.load(config=str(config), match='Other')
        self.assertIsNone(fallback.source_dir)
        self.assertEqual(fallback.game_dir, self.root / 'media/Other')
        self.assertEqual(self.load(config=str(config)).source_dir, cfg.source_dir)

    def test_direct_cli_folder_and_workspace_models(self):
        folder = self.root / 'recording'
        folder.mkdir()
        (self.root / 'models').mkdir()
        model = self.root / 'models/play.pkl'
        model.write_bytes(b'temporary model placeholder')
        config = self.root / 'scout.json'
        Config().save(config)
        cfg = self.load(config=str(config), source_dir=str(folder))
        self.assertEqual(cfg.path(cfg.out_dir), folder / '.scout')
        self.assertEqual(cfg.path(cfg.model_path), model)
        with patch('scout.cli.cmd_scan', return_value=0) as scan:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--source-dir', str(folder), 'scan']), 0)
            self.assertEqual(scan.call_args.args[0].source_dir, str(folder))
        self.assertFalse((self.root / 'out').exists())

    def test_media_workspace_model_is_preserved_for_registry_folder(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        external = self.root / 'external'
        storage.write_library(workspace, {'Game': str(external)})
        (workspace / 'models').mkdir()
        model = workspace / 'models/play.pkl'
        model.write_bytes(b'temporary model placeholder')
        cfg = self.load(match='Game', media_dir=str(workspace / 'media'))
        self.assertEqual(cfg.source_dir, str(external))
        self.assertEqual(cfg.model_path, str(model))

    def test_legacy_saved_paths_adopt_game_storage_and_keep_explicit_output(self):
        config = self.root / 'scout.json'
        game = self.root / 'media/Game'
        for old_out in ('out', 'out/Game', str(self.root / 'out/Game')):
            with self.subTest(old_out=old_out):
                Config(match='Game', out_dir=old_out, fields_dir='fields',
                       cache_dir='.scout-cache').save(config)
                cfg = self.load(config=str(config))
                self.assertEqual(cfg.path(cfg.out_dir), game / '.scout')
                self.assertEqual(cfg.path(cfg.fields_dir), game / '.scout/fields')
                self.assertEqual(cfg.path(cfg.cache_dir), game / '.scout/cache')
                override = self.load(config=str(config), out_dir=old_out)
                self.assertEqual(override.out_dir, old_out)
        Config(match='Game', out_dir='custom/results').save(config)
        self.assertEqual(self.load(config=str(config)).out_dir, 'custom/results')
        self.assertFalse((self.root / 'out').exists())

    def test_invalid_saved_match_is_rejected(self):
        config = self.root / 'scout.json'
        Config(match='../Other').save(config)
        with self.assertRaises(ValueError):
            self.load(config=str(config))

    def test_reselecting_game_rescopes_saved_desktop_paths(self):
        old, new = self.root / 'media/Old', self.root / 'external/New'
        config = self.root / 'scout.json'
        Config(match='Old', source_dir=str(old), media_dir=str(old),
               out_dir=str(old / '.scout'), fields_dir=str(old / '.scout/fields'),
               cache_dir=str(old / '.scout/cache')).save(config)
        storage.write_library(self.root, {'New': str(new)})
        selected = self.load(config=str(config), match='New')
        self.assertEqual(selected.path(selected.out_dir), new / '.scout')
        self.assertEqual(selected.path(selected.fields_dir), new / '.scout/fields')
        self.assertEqual(selected.path(selected.cache_dir), new / '.scout/cache')

    def test_collect_rejects_shared_multisource_review(self):
        state = self.root / '.scout'
        state.mkdir()
        review = state / 'review.mlt'
        review.write_text('<mlt><producer><property name="resource">camera.mp4</property></producer>'
                          '<producer><property name="resource">camera_2.mp4</property></producer></mlt>')
        warnings = []
        self.assertIsNone(collect._gt_path(state, 'camera', warnings))
        self.assertEqual(len(warnings), 1)
        self.assertIn('use cut', warnings[0])

    def test_shared_index_preserves_tombstones_and_rejects_damage(self):
        storage.write_library(self.root, {}, ['Removed'])
        storage.write_library(self.root, {'Game': str(self.root / 'recording')})
        self.assertEqual(storage.read_library(self.root)['removed_legacy'], ['Removed'])
        self.assertEqual(list(self.root.glob('.scout-library-*.tmp')), [])
        (self.root / '.scout-library.json').write_text(json.dumps({'version': 1, 'games': {'../bad': '/tmp'}}))
        with self.assertRaises(ValueError):
            storage.read_library(self.root)
        with self.assertRaises(ValueError):
            self.load(match='bad\\name')

    def test_collect_reads_game_storage_and_excludes_export_sources(self):
        folder = self.root / 'media/Game'
        state = folder / '.scout'
        clips = state / 'clips'
        clips.mkdir(parents=True)
        video = folder / 'camera.mp4'
        video.write_bytes(b'source')
        (clips / 'camera.mp4').write_bytes(b'generated duplicate')
        review = state / 'review.mlt'
        review.write_text('<mlt><producer><property name="resource">camera.mp4</property></producer></mlt>')
        from scout.cut import name_for
        from scout.groundtruth import to_plays
        marks = [Marked(1, 2, 4)]
        exported = clips / name_for(to_plays(marks)[0])
        exported.write_bytes(b'export')
        with patch('scout.collect.probe', return_value=MediaInfo(video, 10, 320, 180, 25, None)), patch('scout.collect.groundtruth.load', return_value=marks):
            rows, warnings = collect.plan(Config(), 'Game', ['camera'], self.root / 'games', self.root / 'media')
        self.assertEqual(warnings, [])
        self.assertEqual(rows[0].src, exported)
        self.assertEqual(rows[0].dst, self.root / 'games/Game/Game - play 001.mp4')


if __name__ == '__main__':
    unittest.main()
