"""Reviewed sources locate their own game state during multi-game operations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scout import cli, detect, pick
from scout.config import Config
from scout.desktop import library
from scout.media import MediaInfo
from scout.timeline import Clip


class SourceStorageTests(unittest.TestCase):
    def test_unscoped_training_reads_each_games_fields_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = Config(media_dir=str(root / 'media'))
            for index, name in enumerate(('A', 'B')):
                folder = root / 'media' / name
                source = folder / 'camera' / 'same.mp4'
                source.parent.mkdir(parents=True)
                source.write_bytes(b'source fixture')
                clip = Clip(MediaInfo(source, 2, 160, 90, 25, None), 0, 0)
                scoped = library.configuration(root, name)
                polygon = [[0, 0], [1, 0], [1, 1]]
                library.save_field(scoped, clip, polygon, 1)
                cached = detect.cache_path(scoped, clip)
                np.savez(cached, offsets=np.array([0, 0]), boxes=np.empty((0, 4)),
                         conf=np.array([]), mask_shape=np.array([90, 160]))
                self.assertEqual(detect.cache_path(cfg, clip), cached)
                self.assertEqual(pick.field_path(cfg, clip).parent, folder / '.scout/fields')
                self.assertEqual(cli._region_for(cfg, [clip], clip)['polygon'], polygon)
                self.assertIsNotNone(detect.load_cached(cfg, clip))
            self.assertFalse((root / 'out').exists())

    def test_custom_cache_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'game' / 'source.mp4'
            source.parent.mkdir()
            source.write_bytes(b'source fixture')
            clip = Clip(MediaInfo(source, 2, 160, 90, 25, None), 0, 0)
            cfg = Config(cache_dir=str(root / 'custom-cache'))
            self.assertEqual(detect.cache_path(cfg, clip).parent, root / 'custom-cache')
