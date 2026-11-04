import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scout import cli, timeline
from scout.config import Config
from scout.media import MediaInfo


class FolderSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "recording"
        self.first = self.video("recording/camera_2.MP4")
        self.second = self.video("recording/nested/camera_10.mov")
        self.video("sibling/camera_1.mp4")
        self.video("recording/Scout/out/export.mp4")
        self.video("recording/nested/Scout/cache/preview.mp4")
        self.cfg = Config(source_dir=str(self.source),
                          media_dir=str(self.root / "absent-media"),
                          match="Display name", stability_seconds=0,
                          out_dir=str(self.source / "Scout/out"))

    def video(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"temporary source fixture")
        return path

    @staticmethod
    def probe(path):
        return MediaInfo(path, 10, 320, 180, 25, None)

    def test_direct_folder_discovery_excludes_siblings_and_application_state(self):
        for alias in ('scout', '.scout'):
            self.video(f'recording/{alias}/clips/export.mp4')
            self.video(f'recording/nested/{alias}/clips/another.mp4')
        self.assertEqual(timeline.discover(self.cfg, wait_for_stable=False),
                         [self.first, self.second])

    def test_cli_and_shared_timeline_have_identical_order_and_offsets(self):
        args = argparse.Namespace(files=None, limit=None)
        with patch("scout.timeline.probe", side_effect=self.probe):
            shared = timeline.build(self.cfg)
            selected = cli._clips(self.cfg, args)
        self.assertEqual([c.path for c in shared], [self.first, self.second])
        self.assertEqual(selected, shared)
        self.assertEqual([c.start for c in selected], [0, 10])

    def test_cli_selection_still_filters_and_rebases_linked_sources(self):
        args = argparse.Namespace(files="camera_10", limit=1)
        with patch("scout.timeline.probe", side_effect=self.probe):
            selected = cli._clips(self.cfg, args)
        self.assertEqual([c.path for c in selected], [self.second])
        self.assertEqual(selected[0].start, 0)

    def test_config_roundtrip_and_cli_load_preserve_explicit_output(self):
        config_path = self.root / "scout.json"
        self.cfg.save(config_path)
        restored = Config.load(config_path)
        self.assertEqual(restored.source_dir, self.cfg.source_dir)
        self.assertEqual(restored.match, "Display name")
        args = argparse.Namespace(config=str(config_path), match=None,
                                  out_dir=None, limit=None)
        loaded = cli._load(args)
        self.assertEqual(loaded.out_dir, self.cfg.out_dir)
        self.assertEqual(loaded.source_dir, self.cfg.source_dir)

    def test_legacy_match_scoping_and_cli_match_output_use_game_storage(self):
        cfg = Config(media_dir=str(self.root), match="sibling", stability_seconds=0)
        self.assertIsNone(cfg.source_dir)
        with patch("scout.timeline.probe", side_effect=self.probe):
            selected = cli._clips(cfg, argparse.Namespace(files=None, limit=None))
        self.assertEqual([c.path for c in selected],
                         [self.root / "sibling/camera_1.mp4"])
        args = argparse.Namespace(config=None, match="sibling", out_dir=None,
                                  limit=None)
        self.assertEqual(cli._load(args).out_dir, "Scout")
        args.out_dir = str(self.root / "custom-output")
        self.assertEqual(cli._load(args).out_dir, args.out_dir)

    def test_legacy_unscoped_discovery_keeps_recursive_filename_order(self):
        cfg = Config(media_dir=str(self.root), stability_seconds=0)
        self.assertEqual(timeline.discover(cfg, wait_for_stable=False),
                         [self.root / "sibling/camera_1.mp4", self.first, self.second])


if __name__ == "__main__":
    unittest.main()
