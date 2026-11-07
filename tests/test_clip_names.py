"""Pattern expansion and temporary export publication regressions."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scout import cli, cut
from scout.clip_names import pattern_names
from scout.config import Config


class PatternNamesTests(unittest.TestCase):
    def test_position_width_extension_and_zero(self):
        self.assertEqual(pattern_names(2, '<N>_review', 9),
                         ['09_review.mp4', '10_review.mp4'])
        self.assertEqual(pattern_names(2, 'game_<N>.MP4', 0, 3),
                         ['game_000.MP4', 'game_001.MP4'])
        self.assertEqual(pattern_names(2, 'clip<N>', 100, 0),
                         ['clip100.mp4', 'clip101.mp4'])
        self.assertEqual(pattern_names(0, 'clip<N>'), [])
        self.assertEqual(pattern_names(10, '<N>', 0)[0], '00.mp4')

    def test_invalid_patterns_and_numbers(self):
        for pattern in ('missing', '<N><N>', '../<N>', 'folder\\<N>',
                        '<N>\n', '<N>\x00', '<N>:', '<N>.', '<N> ',
                        'a' * 255 + '<N>', 'CON.<N>'):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                pattern_names(1, pattern)
        for kwargs in ({'count': -1}, {'count': 1.5}, {'start': -1},
                       {'start': True}, {'pad': -1}, {'pad': '2'}, {'pad': 1000000}):
            args = dict(count=1, pattern='<N>')
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pattern_names(**args)


class CustomExportTests(unittest.TestCase):
    def test_queued_review_change_is_rejected_before_encoding(self):
        source = self.out/'review.mlt'
        source.write_text('changed after the export was queued')
        with patch.object(cut, 'cut_all') as render, patch.object(cli.groundtruth, 'project_clips') as probe:
            self.assertEqual(cli.main(['cut', '--mlt', str(source), '--name-pattern', 'game_<N>',
                                      '--review-hash', 'outdated preview']), 2)
            render.assert_not_called()
            probe.assert_not_called()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.cfg = Config(clip_workers=2, source_dir=self.directory.name)
        # Deliberately nonconsecutive play indices: selected plays are numbered
        # in their supplied chronological order, independently of marker IDs.
        self.plays = [SimpleNamespace(index=4, snap=10, start=10, end=20),
                      SimpleNamespace(index=9, snap=30, start=30, end=40)]
        self.clips = [SimpleNamespace(end=60)]

    def render(self, cfg, clips, play, target, **kwargs):
        target.write_text(str(play.index))
        return target

    def export(self, plays=None, **kwargs):
        return cut.cut_all(self.cfg, self.clips, self.plays if plays is None else plays,
                           self.out, progress=False, **kwargs)

    def manifest(self):
        return json.loads((self.out / 'exports.json').read_text())

    def test_custom_export_reexport_and_partial(self):
        (self.out / 'personal.mp4').write_text('keep')
        with patch.object(cut, 'cut_one', side_effect=self.render):
            self.export(name_pattern='match_<N>', start_number=9)
            self.assertEqual(self.manifest(), ['match_09.mp4', 'match_10.mp4'])
            self.assertEqual((self.out / 'match_09.mp4').read_text(), '4')
            self.export(plays=self.plays[1:], name_pattern='<N>_selected',
                        start_number=42, partial=True)
            self.assertEqual(self.manifest(),
                             ['42_selected.mp4', 'match_09.mp4', 'match_10.mp4'])
            self.export(plays=self.plays[:1], name_pattern='match_<N>',
                        start_number=9, number_width=2)
        self.assertEqual(self.manifest(), ['match_09.mp4'])
        self.assertFalse((self.out / 'match_10.mp4').exists())
        self.assertFalse((self.out / '42_selected.mp4').exists())
        self.assertEqual((self.out / 'personal.mp4').read_text(), 'keep')

    def test_legacy_default_names(self):
        with patch.object(cut, 'cut_one', side_effect=self.render):
            paths = self.export()
        self.assertEqual([p.name for p in paths],
                         ['play_004_00m10s.mp4', 'play_009_00m30s.mp4'])
        self.assertEqual(self.manifest(), [p.name for p in paths])

    def test_invalid_and_existing_names_fail_before_encoding(self):
        target = self.out / 'game_1.mp4'
        target.write_text('personal')
        with patch.object(cut, 'cut_one') as render:
            for pattern in ('../<N>', 'game_<N>'):
                with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                    self.export(name_pattern=pattern)
            render.assert_not_called()
        self.assertEqual(target.read_text(), 'personal')
        self.assertFalse((self.out / 'exports.json').exists())

    def test_collision_appearing_during_render_preserves_previous_batch(self):
        old = self.out / 'old.mp4'
        old.write_text('old')
        (self.out / 'exports.json').write_text('["old.mp4"]')

        def render(cfg, clips, play, target, **kwargs):
            self.render(cfg, clips, play, target)
            if play.index == 9:
                (self.out / 'game_2.mp4').write_text('arrived')
            return target

        with patch.object(cut, 'cut_one', side_effect=render):
            with self.assertRaises(ValueError):
                self.export(name_pattern='game_<N>')
        self.assertEqual(old.read_text(), 'old')
        self.assertEqual(self.manifest(), ['old.mp4'])
        self.assertEqual((self.out / 'game_2.mp4').read_text(), 'arrived')
        self.assertFalse((self.out / 'game_1.mp4').exists())
        self.assertFalse(list(self.out.glob('.export-*')))

    def test_collision_at_publication_does_not_overwrite_or_leave_claims(self):
        original_link = Path.hardlink_to

        def link(target, source):
            if target.name == 'game_2.mp4':
                target.write_text('arrived at publication')
            return original_link(target, source)

        with patch.object(cut, 'cut_one', side_effect=self.render), \
                patch.object(Path, 'hardlink_to', new=link):
            with self.assertRaises(FileExistsError):
                self.export(name_pattern='game_<N>')
        self.assertEqual((self.out / 'game_2.mp4').read_text(), 'arrived at publication')
        self.assertFalse((self.out / 'game_1.mp4').exists())
        self.assertFalse((self.out / 'exports.json').exists())

    def test_manifest_owned_symlink_is_not_replaced(self):
        personal = self.out / 'personal.mp4'
        personal.write_text('keep')
        (self.out / 'game_1.mp4').symlink_to(personal)
        (self.out / 'exports.json').write_text('["game_1.mp4"]')
        with patch.object(cut, 'cut_one') as render:
            with self.assertRaises(ValueError):
                self.export(name_pattern='game_<N>')
            render.assert_not_called()
        self.assertEqual(personal.read_text(), 'keep')
        self.assertTrue((self.out / 'game_1.mp4').is_symlink())

    def test_cli_passes_pattern_options(self):
        with patch.object(cli, '_load', return_value=self.cfg), \
                patch.object(cli.groundtruth, 'project_clips', return_value=[]), \
                patch.object(cli.groundtruth, 'load', return_value=[]), \
                patch.object(cli.groundtruth, 'to_plays', return_value=self.plays), \
                patch.object(cli.cut, 'cut_all', return_value=[]) as export:
            self.assertEqual(cli.main(['cut', '--name-pattern', '<N>_game',
                                      '--start-number', '7', '--number-width', '3',
                                      '--only', '9']), 0)
        self.assertEqual(export.call_args.args[2], [self.plays[1]])
        self.assertEqual(export.call_args.kwargs, dict(partial=True,
                         name_pattern='<N>_game', start_number=7, number_width=3,
                         pre_play=None, post_play=None))
