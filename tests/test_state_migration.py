"""Migration safety uses temporary game folders, never a real library."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from scout import state_migration as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def game(self, name='game', legacy='.scout'):
        game = self.root / name
        state = game / legacy
        state.mkdir(parents=True)
        (game / 'camera.mp4').write_bytes(b'footage')
        (state / 'cache').mkdir()
        (state / 'cache' / 'detection.npz').write_bytes(b'cached detections')
        return game, state

    def test_review_resources_markers_cache_and_backups(self):
        game, state = self.game()
        # Both relative outside footage and absolute paths inside state.
        review = ('<?xml version="1.0"?><mlt><!-- preserve -->'
                  '<producer id="a"><property name="resource">../camera.mp4</property></producer>'
                  f'<producer id="b"><property name="resource">{state}/proxy.mp4</property></producer>'
                  '<properties name="shotcut:markers"><properties><property name="start">10</property>'
                  '<property name="end">10</property><property name="text">unsaved note .scout</property>'
                  '</properties></properties><custom foo="keep"/></mlt>').encode()
        (state / 'review.mlt').write_bytes(review)
        (state / 'paths.json').write_text(json.dumps({'sources': [str(game / 'camera.mp4')],
            'cache': str(state / 'cache'), 'field_dir': '.scout/fields', 'untouched': '.scout-cache/x'}))
        before = [(p, p.stat()) for p in [game / 'camera.mp4', state / 'cache' / 'detection.npz']]
        plan = migration.plan_migration([game])
        self.assertTrue(state.exists())
        backup = self.root / 'backup'
        result = migration.apply_migration(plan, backup)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(state.exists())
        target = game / 'Scout'
        self.assertEqual((backup / '0' / 'review.mlt').read_bytes(), review)
        xml = ET.parse(target / 'review.mlt').getroot()
        self.assertEqual(xml.get('root'), str(target))
        self.assertEqual((Path(xml.get('root')) / xml.find('producer/property').text).resolve(), game / 'camera.mp4')
        self.assertEqual(xml.find("properties/properties/property[@name='text']").text, 'unsaved note .scout')
        self.assertIn(b'<!-- preserve -->', (target / 'review.mlt').read_bytes())
        data = json.loads((target / 'paths.json').read_text())
        self.assertEqual(data['field_dir'], 'Scout/fields')
        self.assertEqual(data['untouched'], '.scout-cache/x')
        for path, info in before:
            new = target / path.relative_to(state) if path.is_relative_to(state) else path
            after = new.stat()
            self.assertEqual((info.st_ino, info.st_size, info.st_mtime_ns),
                             (after.st_ino, after.st_size, after.st_mtime_ns))

    def test_timewarp_and_file_uri_resources(self):
        game, state = self.game()
        old = str(state / 'proxy clip.mp4')
        uri = (state / 'proxy clip.mp4').as_uri()
        (state / 'review.mlt').write_text(
            f'<mlt root="{state}"><!-- intact -->'
            f'<producer><property name="resource">0.5:{old}</property>'
            f'<property name="warp_resource">{old}</property></producer>'
            f'<producer><property name="resource">{uri}</property></producer>'
            '<property name="text">.scout/play</property></mlt>')
        migration.apply_migration(migration.plan_migration([game]), self.root / 'backup')
        data = (game / 'Scout' / 'review.mlt').read_text()
        root = ET.fromstring(data)
        resources = [p.text for p in root.findall('producer/property')]
        self.assertEqual(resources, ['0.5:' + str(game / 'Scout' / 'proxy clip.mp4'),
                                    str(game / 'Scout' / 'proxy clip.mp4'),
                                    (game / 'Scout' / 'proxy clip.mp4').as_uri()])
        self.assertIn('<!-- intact -->', data)
        self.assertEqual(root.find('property').text, '.scout/play')

    def test_windows_paths_resources_and_json_annotations(self):
        mappings = [(r'C:\game\.scout', r'C:\game\Scout')]
        xml = (r'<mlt root="C:\game\.scout"><producer>'
               r'<property name="resource">0.5:C:\game\.scout\proxy.mp4</property>'
               r'<property name="warp_resource">C:/game/.scout/proxy.mp4</property>'
               r'<property name="shotcut:projectFolder">file:///C:/game/.scout</property>'
               r'</producer><property name="text">scout</property></mlt>')
        root = ET.fromstring(migration._metadata(Path('review.mlt'), xml.encode(), mappings))
        self.assertEqual(root.get('root'), r'C:\game\Scout')
        self.assertEqual([p.text for p in root.findall('producer/property')],
                         [r'0.5:C:\game\Scout\proxy.mp4', r'C:\game\Scout\proxy.mp4',
                          'file:///C:/game/Scout'])
        original = {'name': 'scout', 'label': '.scout/play', 'notes': ['scout', '.scout/play'],
                    '.scout/play': 'unchanged key', 'path': r'.scout\cache',
                    'cache_dir': r'.\scout\cache', 'absolute': r'C:\game\.scout\cache',
                    'other': '.scout/play', 'cache_identity': r'C:\game\.scout-cache\x'}
        updated = json.loads(migration._metadata(Path('paths.json'), json.dumps(original).encode(), mappings))
        for key in ['name', 'label', 'notes', '.scout/play', 'other', 'cache_identity']:
            self.assertEqual(updated[key], original[key])
        self.assertEqual(updated['path'], r'Scout\cache')
        self.assertEqual(updated['cache_dir'], r'.\Scout\cache')
        self.assertEqual(updated['absolute'], r'C:\game\Scout\cache')

    def test_lowercase_legacy(self):
        game, state = self.game(legacy='scout')
        (state / 'paths.json').write_text(json.dumps({'path': 'scout/cache'}))
        migration.apply_migration(migration.plan_migration([game]), self.root / 'backup')
        self.assertEqual(json.loads((game / 'Scout' / 'paths.json').read_text())['path'], 'Scout/cache')

    def test_windows_path_casing_preserves_the_same_reference(self):
        mappings = [(r'C:\Games\one\.scout', r'C:\Games\one\Scout')]
        data = json.dumps({'cache_dir': r'c:\games\ONE\.SCOUT\cache'}).encode()
        updated = json.loads(migration._metadata(Path('config.json'), data, mappings))
        self.assertEqual(updated['cache_dir'], r'C:\Games\one\Scout\cache')
        xml = (r'<mlt root="c:\games\ONE\.SCOUT"><producer>'
               r'<property name="resource">0.5:c:\GAMES\one\.SCOUT\proxy.mp4</property>'
               r'</producer></mlt>')
        updated = ET.fromstring(migration._metadata(Path('review.mlt'), xml.encode(), mappings))
        self.assertEqual(updated.get('root'), r'C:\Games\one\Scout')
        self.assertEqual(updated.find('producer/property').text, r'0.5:C:\Games\one\Scout\proxy.mp4')

    def test_manually_renamed_state_rebases_both_historical_path_prefixes(self):
        game, state = self.game(legacy='scout')
        hidden = game / '.scout'
        (state / 'paths.json').write_text(json.dumps({'cache_dir': str(hidden / 'cache')}))
        (state / 'review.mlt').write_text(f'<mlt root="{hidden}"><producer>'
            '<property name="resource">../camera.mp4</property></producer></mlt>')
        migration.apply_migration(migration.plan_migration([game]), self.root / 'backup')
        new = game / 'Scout'
        self.assertEqual(json.loads((new / 'paths.json').read_text())['cache_dir'], str(new / 'cache'))
        root = ET.parse(new / 'review.mlt').getroot()
        self.assertEqual(root.get('root'), str(new))
        self.assertEqual((Path(root.get('root')) / root.find('producer/property').text).resolve(), game / 'camera.mp4')

    def test_collisions_and_links_do_not_mutate(self):
        game, state = self.game()
        (game / 'Scout').mkdir()
        with self.assertRaises(migration.MigrationError):
            migration.plan_migration([game])
        self.assertTrue(state.exists())
        (game / 'Scout').rmdir()
        (state / 'linked').symlink_to(game / 'camera.mp4')
        with self.assertRaises(migration.MigrationError):
            migration.plan_migration([game])

    def test_plan_conflict_refused_before_backup(self):
        game, state = self.game()
        plan = migration.plan_migration([game])
        (state / 'new.txt').write_text('new note')
        with self.assertRaises(migration.MigrationError):
            migration.apply_migration(plan, self.root / 'backup')
        self.assertFalse((self.root / 'backup').exists())
        self.assertTrue(state.exists())

    def test_python_package_refused(self):
        game, state = self.game(legacy='scout')
        (state / '__init__.py').write_text('')
        with self.assertRaises(migration.MigrationError):
            migration.plan_migration([game])

    def test_multiple_legacy_folders_refused(self):
        game, state = self.game()
        (game / 'scout').mkdir()
        with self.assertRaises(migration.MigrationError):
            migration.plan_migration([game])
        self.assertTrue(state.exists())

    def test_relative_root_and_nested_review_resolution(self):
        game, state = self.game()
        review_dir = state / 'reviews'
        review_dir.mkdir()
        (review_dir / 'review.mlt').write_text(
            '<mlt root="../.."><producer><property name="resource">camera.mp4</property>'
            '</producer></mlt>')
        migration.apply_migration(migration.plan_migration([game]), self.root / 'backup')
        root = ET.parse(game / 'Scout' / 'reviews' / 'review.mlt').getroot()
        self.assertEqual((Path(root.get('root')) / root.find('producer/property').text).resolve(),
                         game / 'camera.mp4')

    def test_live_batch_refused(self):
        game, state = self.game()
        batch = state / 'desktop' / 'batches' / 'one'
        batch.mkdir(parents=True)
        (batch / 'pid').write_text(str(os.getpid()))
        with self.assertRaises(migration.MigrationError):
            migration.plan_migration([game])

    def test_whole_batch_rollback_after_metadata_failure(self):
        games = []
        originals = []
        for name in ['one', 'two']:
            game, state = self.game(name)
            original = json.dumps({'path': str(state / 'cache')}).encode()
            (state / 'paths.json').write_bytes(original)
            games.append(game)
            originals.append(original)
        plan = migration.plan_migration(games)
        atomic = migration._atomic
        failures = []
        def fail_once(path, data, mode=0o600):
            if path == games[1] / 'Scout' / 'paths.json' and not failures:
                failures.append(True)
                raise OSError('injected write failure')
            return atomic(path, data, mode)
        with patch.object(migration, '_atomic', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, 'injected'):
                migration.apply_migration(plan, self.root / 'backup')
        for game, original in zip(games, originals):
            self.assertFalse((game / 'Scout').exists())
            self.assertEqual((game / '.scout' / 'paths.json').read_bytes(), original)
        self.assertEqual(json.loads((self.root / 'backup' / 'result.json').read_text())['status'], 'rolled_back')

    def test_backup_inside_state_refused(self):
        game, state = self.game()
        with self.assertRaises(migration.MigrationError):
            migration.apply_migration(migration.plan_migration([game]), state / 'backup')


if __name__ == '__main__':
    unittest.main()
