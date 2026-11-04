"""Desktop operations must preserve source footage and authoritative reviews."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scout.desktop import library
from scout.media import MediaInfo
from scout.timeline import Clip


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_match_names_cannot_escape_library(self):
        for name in ('', '.', '..', '../other', '/tmp/match', 'a/b', 'a\\b', 'a\nb'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                library.match_path(self.root, name)
        self.assertEqual(library.match_path(self.root, 'SAI vs HON'), self.root/'media/SAI vs HON')

    def test_symlink_outside_library_is_rejected(self):
        (self.root/'media').mkdir()
        (self.root/'media/escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            library.match_path(self.root, 'escape')

    def test_native_field_is_readable_by_existing_cli(self):
        from scout.cli import _region_for
        cfg = library.configuration(self.root, 'game')
        source = self.root/'media/game/camera.MP4'
        clip = Clip(MediaInfo(source, 120., 1920, 1080, 30., None), 0., 0)
        polygon = [[.1, .2], [.9, .2], [.9, .8], [.1, .8]]
        library.save_field(cfg, clip, polygon, 30.)
        region = _region_for(cfg, [clip], clip)
        self.assertEqual(region['polygon'], polygon)
        self.assertEqual(region['ref_time'], 30.)
        with self.assertRaises(ValueError):
            library.save_field(cfg, clip, polygon, 120.)
        self.assertEqual(_region_for(cfg, [clip], clip)['ref_time'], 30.)

    def test_remove_match_preserves_resources_and_can_be_restored(self):
        files = ['media/game/camera.mp4', 'out/game/review.mlt',
                 'out/game/clip.mp4', 'fields/game.json', '.scout-cache/detection.json']
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'preserve exactly')
        (self.root / 'media/other').mkdir()
        library.remove_match(self.root, 'game')
        self.assertEqual([r['name'] for r in library.matches(self.root)], ['other'])
        for name in files:
            self.assertEqual((self.root / name).read_bytes(), b'preserve exactly')
        with self.assertRaises(FileExistsError):
            library.remove_match(self.root, 'game')
        (self.root / 'media/game/.scout-removed').unlink()
        self.assertEqual([r['name'] for r in library.matches(self.root)], ['game', 'other'])

    def test_remove_match_rejects_missing_and_unsafe_paths(self):
        for name in ('missing', '../outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                library.remove_match(self.root, name)
        (self.root / 'media').mkdir()
        (self.root / 'media/escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            library.remove_match(self.root, 'escape')

    def test_folder_import_persists_without_copying_and_routes_game_data(self):
        from scout import detect
        game = self.root / 'external' / 'Away game'
        game.mkdir(parents=True)
        source = game / 'camera.mp4'
        source.write_bytes(b'original footage')
        state = game / 'Scout'
        (state / 'clips').mkdir(parents=True)
        (state / 'clips/export.mp4').write_bytes(b'generated clip')
        root = self.root / 'library'
        self.assertEqual(library.import_folder(root, game), 'Away game')
        self.assertEqual(library.import_folder(root, game), 'Away game')
        # Re-reading from disk simulates a fresh process with no in-memory index.
        self.assertEqual(library.linked_games(root), {'Away game': str(game)})
        self.assertEqual(library.matches(root)[0]['files'], 1)
        self.assertFalse((root / 'media').exists())
        cfg = library.configuration(root, 'Away game')
        self.assertEqual(cfg.source_dir, str(game))
        self.assertEqual(cfg.root, game)
        self.assertEqual(cfg.path(cfg.out_dir), state)
        self.assertEqual(cfg.path(cfg.fields_dir), state / 'fields')
        self.assertEqual(cfg.path(cfg.cache_dir), state / 'cache')
        clip = Clip(MediaInfo(source, 10, 320, 180, 25, None), 0, 0)
        library.save_field(cfg, clip, [[0, 0], [1, 0], [1, 1]], 1)
        self.assertTrue(list((state / 'fields').glob('*.json')))
        self.assertEqual(detect.cache_path(cfg, clip).parent, state / 'cache')
        with patch('scout.timeline.probe', return_value=clip.info):
            loaded = library.load_match(root, 'Away game')
        self.assertEqual([c.path for c in loaded['clips']], [source])
        self.assertEqual(loaded['review'], state / 'review.mlt')
        self.assertTrue(all(loaded['regions']))
        self.assertEqual(source.read_bytes(), b'original footage')
        with self.assertRaises(ValueError):
            library.import_videos(root, 'Away game', [source])

    def test_linked_game_removal_and_reimport_preserve_saved_work(self):
        game = self.root / 'external' / 'Game'
        game.mkdir(parents=True)
        (game / 'source.mp4').write_bytes(b'original')
        root = self.root / 'library'
        library.import_folder(root, game)
        review = game / 'Scout/review.mlt'
        review.write_bytes(b'user edited review')
        library.remove_match(root, 'Game')
        self.assertEqual(library.matches(root), [])
        self.assertEqual(review.read_bytes(), b'user edited review')
        self.assertFalse((game / '.scout-removed').exists())
        library.import_folder(root, game)
        self.assertTrue(library.matches(root)[0]['review'])
        self.assertEqual(review.read_bytes(), b'user edited review')

    def test_empty_folder_import_persists_without_counting_generated_videos(self):
        root = self.root / 'library'
        game = self.root / 'Upcoming game'
        game.mkdir()
        self.assertEqual(library.import_folder(root, game), game.name)
        self.assertEqual(library.linked_games(root), {game.name: str(game)})
        exported = game / 'Scout/clips/export.mp4'
        exported.parent.mkdir()
        exported.write_bytes(b'generated video')
        self.assertEqual(library.matches(root)[0]['files'], 0)
        data = library.load_match(root, game.name)
        self.assertEqual(data['clips'], [])
        self.assertEqual(data['regions'], [])
        source = game / 'camera.mp4'
        source.write_bytes(b'camera fixture')
        with patch('scout.timeline.probe', return_value=MediaInfo(source, 10, 320, 180, 25, None)):
            loaded = library.load_match(root, game.name)
        self.assertEqual(library.matches(root)[0]['files'], 1)
        self.assertEqual([clip.path for clip in loaded['clips']], [source])

    def test_folder_import_rejects_collisions_and_external_state(self):
        root = self.root / 'library'
        first, second = self.root / 'a/Game', self.root / 'b/Game'
        for game in (first, second):
            game.mkdir(parents=True)
            (game / 'source.mp4').write_bytes(b'original')
        library.import_folder(root, first)
        with self.assertRaises(ValueError):
            library.import_folder(root, second)
        self.assertEqual(library.linked_games(root), {'Game': str(first)})
        legacy = self.root / 'external/Legacy'
        legacy.mkdir(parents=True)
        (legacy / 'source.mp4').write_bytes(b'original')
        (root / 'media/Legacy').mkdir(parents=True)
        with self.assertRaises(ValueError):
            library.import_folder(root, legacy)
        external = self.root / 'state elsewhere'
        external.mkdir()
        unsafe = self.root / 'Unsafe'
        unsafe.mkdir()
        (unsafe / 'source.mp4').write_bytes(b'original')
        (unsafe / 'Scout').symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            library.import_folder(root, unsafe)
        self.assertEqual(list(external.iterdir()), [])

    def test_offline_game_stays_indexed_and_can_be_removed_without_recreating_folder(self):
        game = self.root / 'drive/Game'
        game.mkdir(parents=True)
        (game / 'source.mp4').write_bytes(b'original')
        root = self.root / 'library'
        library.import_folder(root, game)
        moved = game.with_name('unavailable')
        game.rename(moved)
        row, = library.matches(root)
        self.assertEqual(row['name'], 'Game')
        self.assertFalse(row['available'])
        with self.assertRaises(ValueError):
            library.load_match(root, 'Game')
        library.remove_match(root, 'Game')
        self.assertFalse(game.exists())
        self.assertEqual(library.matches(root), [])
        self.assertEqual((moved / 'source.mp4').read_bytes(), b'original')

    def test_invalid_index_and_failed_atomic_save_preserve_existing_index(self):
        root = self.root / 'library'
        root.mkdir()
        index = root / '.scout-library.json'
        index.write_text('{invalid JSON')
        game = self.root / 'Game'
        game.mkdir()
        (game / 'source.mp4').write_bytes(b'original')
        with self.assertRaises(ValueError):
            library.import_folder(root, game)
        self.assertEqual(index.read_text(), '{invalid JSON')
        index.unlink()
        library.import_folder(root, game)
        original = index.read_bytes()
        with patch('scout.desktop.library.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                library.remove_match(root, 'Game')
        self.assertEqual(index.read_bytes(), original)
        self.assertEqual(list(root.glob('.scout-library-*.tmp')), [])

    def legacy_fixture(self):
        from scout import groundtruth, shotcut
        game = self.root / 'media/Game'
        game.mkdir(parents=True)
        source = game / 'camera.mp4'
        source.write_bytes(b'original footage')
        clip = Clip(MediaInfo(source, 10, 320, 180, 25, None), 0, 0)
        from scout.config import Config
        cfg = Config(media_dir=str(self.root / 'media'), match='Game',
                     out_dir=str(self.root / 'out/Game'),
                     fields_dir=str(self.root / 'fields'), cache_dir=str(self.root / '.scout-cache'))
        library.save_field(cfg, clip, [[0, 0], [1, 0], [1, 1]], 1)
        review = self.root / 'out/Game/review.mlt'
        review.parent.mkdir(parents=True)
        shotcut.write_review(cfg, [clip], groundtruth.to_plays([groundtruth.Marked(1, 1, 3)]), review)
        cache = self.root / '.scout-cache/camera.saved.npz'
        cache.parent.mkdir()
        cache.write_bytes(b'saved detections')
        exported = review.parent / 'clips/old.mp4'
        exported.parent.mkdir()
        exported.write_bytes(b'previous export')
        return game, clip, review, cache, exported

    def test_previous_match_adopts_saved_work_without_moving_or_overwriting(self):
        from scout import pick
        game, clip, review, cache, exported = self.legacy_fixture()
        original = review.read_bytes()
        self.assertEqual(library.import_folder(self.root, game), 'Game')
        cfg = library.configuration(self.root, 'Game')
        self.assertEqual(cfg.source_dir, str(game))
        self.assertEqual((game / 'Scout/review.mlt').read_bytes(), original)
        self.assertEqual((game / 'Scout/cache' / cache.name).read_bytes(), cache.read_bytes())
        self.assertTrue(pick.field_path(cfg, clip).is_file())
        self.assertEqual((game / 'Scout/clips/old.mp4').read_bytes(), exported.read_bytes())
        with patch('scout.timeline.probe', return_value=clip.info), patch('scout.media.probe', return_value=clip.info):
            loaded = library.load_match(self.root, 'Game')
        self.assertEqual(len(loaded['plays']), 1)
        self.assertTrue(all(loaded['regions']))
        self.assertEqual([c.path for c in loaded['clips']], [clip.path])
        (game / 'Scout/review.mlt').write_bytes(b'newer game review')
        (game / 'Scout/cache' / cache.name).write_bytes(b'newer cache')
        self.assertEqual(review.read_bytes(), original)
        self.assertEqual(cache.read_bytes(), b'saved detections')
        library.remove_match(self.root, 'Game')
        self.assertEqual(library.matches(self.root), [])
        library.import_folder(self.root, game)
        self.assertEqual((game / 'Scout/review.mlt').read_bytes(), b'newer game review')
        self.assertEqual(review.read_bytes(), original)

    def test_previous_match_relative_review_keeps_original_source_base(self):
        from scout import groundtruth
        game, clip, review, _, _ = self.legacy_fixture()
        review.write_text(review.read_text().replace(str(clip.path), '../../media/Game/camera.mp4'))
        original = review.read_bytes()
        library.import_folder(self.root, game)
        with patch('scout.media.probe', return_value=clip.info):
            self.assertEqual([c.path for c in groundtruth.project_clips(game / 'Scout/review.mlt')], [clip.path])
        self.assertEqual(review.read_bytes(), original)

    def test_previous_match_migration_failure_leaves_originals_and_can_retry(self):
        game, _, review, _, _ = self.legacy_fixture()
        original = review.read_bytes()
        with patch('scout.desktop.library.shutil.copy2', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                library.import_folder(self.root, game)
        self.assertEqual(library.linked_games(self.root), {})
        self.assertEqual(review.read_bytes(), original)
        library.import_folder(self.root, game)
        self.assertEqual((game / 'Scout/review.mlt').read_bytes(), original)

    def test_previous_match_migration_supports_filesystems_without_hardlinks(self):
        game, _, review, _, exported = self.legacy_fixture()
        with patch('scout.desktop.library.os.link', side_effect=OSError('no hardlinks')):
            library.import_folder(self.root, game)
        self.assertEqual((game / 'Scout/review.mlt').read_bytes(), review.read_bytes())
        self.assertEqual((game / 'Scout/clips/old.mp4').read_bytes(), exported.read_bytes())

    def test_previous_match_unregister_failure_preserves_visibility(self):
        game, _, _, _, _ = self.legacy_fixture()
        library.import_folder(self.root, game)
        with patch('scout.desktop.library.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                library.remove_match(self.root, 'Game')
        self.assertFalse((game / '.scout-removed').exists())
        self.assertEqual([r['name'] for r in library.matches(self.root)], ['Game'])

    def test_offline_previous_match_stays_removed_after_reconnection(self):
        game, _, review, _, _ = self.legacy_fixture()
        original = review.read_bytes()
        library.import_folder(self.root, game)
        offline = game.with_name('disconnected')
        game.rename(offline)
        library.remove_match(self.root, 'Game')
        offline.rename(game)
        self.assertEqual(library.matches(self.root), [])
        library.import_folder(self.root, game)
        self.assertTrue(library.matches(self.root)[0]['linked'])
        self.assertEqual((game / 'Scout/review.mlt').read_bytes(), original)

    def test_import_copies_without_touching_originals(self):
        source = self.root/'camera.MP4'
        source.write_bytes(b'video')
        progress = []
        with patch('scout.media.probe'):
            count = library.import_videos(self.root, 'game', [source], progress.append)
        self.assertEqual(count, 1)
        self.assertEqual((self.root/'media/game/camera.MP4').read_bytes(), b'video')
        self.assertEqual(source.read_bytes(), b'video')
        self.assertEqual(progress[-1], 100)

    def test_duplicate_import_leaves_existing_footage_intact(self):
        dest = self.root/'media/game'
        dest.mkdir(parents=True)
        (dest/'camera.mp4').write_bytes(b'original')
        source = self.root/'CAMERA.MP4'
        source.write_bytes(b'replacement')
        with self.assertRaises(ValueError):
            library.import_videos(self.root, 'game', [source])
        self.assertEqual((dest/'camera.mp4').read_bytes(), b'original')

    def test_failed_import_rolls_back_committed_files(self):
        sources = [self.root/'a.mp4', self.root/'b.mp4']
        for p in sources:
            p.write_bytes(b'video')
        import os
        link = os.link
        def fail_second(src, dst):
            if Path(dst).name == 'b.mp4':
                raise OSError('simulated failure')
            link(src, dst)
        with patch('scout.media.probe'), patch('scout.desktop.library.os.link', side_effect=fail_second):
            with self.assertRaises(OSError):
                library.import_videos(self.root, 'game', sources)
        self.assertEqual(list((self.root/'media/game').iterdir()), [])
        self.assertTrue(all(p.exists() for p in sources))

    def test_unreadable_import_is_rejected_before_copying(self):
        source = self.root/'broken.mp4'
        source.write_bytes(b'broken')
        with patch('scout.media.probe', side_effect=ValueError('broken video')):
            with self.assertRaises(ValueError):
                library.import_videos(self.root, 'game', [source])
        self.assertFalse((self.root/'media/game').exists())

    def test_low_disk_space_rejects_import(self):
        from collections import namedtuple
        source = self.root/'video.mp4'
        source.write_bytes(b'video')
        usage = namedtuple('usage', 'total used free')(100, 99, 1)
        with patch('scout.media.probe'), patch('scout.desktop.library.shutil.disk_usage', return_value=usage):
            with self.assertRaisesRegex(ValueError, 'disk space'):
                library.import_videos(self.root, 'game', [source])

    def test_running_batch_is_detected_but_stale_batch_is_ignored(self):
        import os
        batch = self.root/'Scout/batches/w3-batch-test'
        batch.mkdir(parents=True)
        (batch/'pid').write_text(str(os.getpid()))
        (batch/'status.json').write_text(json.dumps({'A': {'state': 'running'}, 'B': {'state': 'completed'}}))
        self.assertEqual([r['name'] for r in library.active_batches(self.root)], ['A'])
        with patch('scout.desktop.library.os.kill', side_effect=ProcessLookupError):
            self.assertEqual(library.active_batches(self.root), [])

    def make_review(self):
        from scout import shotcut, groundtruth
        source = self.root/'camera.mp4'
        clip = Clip(MediaInfo(source, 10., 160, 90, 25., None), 0., 0)
        review = self.root/'review.mlt'
        shotcut.write_review(library.configuration(self.root, 'game'), [clip],
            groundtruth.to_plays([groundtruth.Marked(1, 1., 3., 'Original')]), review)
        return review

    def test_marker_edit_preserves_sources_and_point_notes_with_backup(self):
        import hashlib
        import xml.etree.ElementTree as ET
        from scout import groundtruth
        review = self.make_review()
        tree = ET.parse(review)
        block = tree.find(".//properties[@name='shotcut:markers']")
        note = ET.SubElement(block, 'properties', name='1')
        for key, value in {'start': '00:00:05.000', 'end': '00:00:05.000', 'text': 'Timeout'}.items():
            ET.SubElement(note, 'property', name=key).text = value
        tree.write(review)
        original = review.read_bytes()
        source_xml = ET.tostring(ET.parse(review).find("playlist[@id='playlist0']"))
        digest = library.save_markers(review, [groundtruth.Marked(1, 2., 4., 'Edited')], hashlib.sha256(original).hexdigest())
        markers = groundtruth.load(review)
        self.assertEqual((markers[0].start, markers[0].end, markers[0].text), (2., 4., 'Edited'))
        self.assertEqual(ET.tostring(ET.parse(review).find("playlist[@id='playlist0']")), source_xml)
        self.assertIn('Timeout', review.read_text())
        self.assertEqual(review.with_name('review.before-studio.mlt').read_bytes(), original)
        library.save_markers(review, [], digest)
        self.assertEqual(groundtruth.load(review), [])
        self.assertIn('Timeout', review.read_text())

    def test_marker_edit_rejects_external_change_and_bad_bounds(self):
        import hashlib
        from scout.groundtruth import Marked
        review = self.make_review()
        before = review.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        with self.assertRaisesRegex(ValueError, 'changed outside'):
            library.save_markers(review, [], 'outdated-hash')
        for a, b in [(4, 2), (-1, 2), (1, 11), (float('nan'), 2)]:
            with self.assertRaises(ValueError):
                library.save_markers(review, [Marked(1, a, b)], digest)
        self.assertEqual(review.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
