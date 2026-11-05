"""Library-index migration changes registry data only and preserves conflicts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scout.storage import import_library_index, index_digest, read_library, write_library


class IndexImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source, self.target = self.base/'old', self.base/'new'

    def test_merge_backs_up_exact_original_and_does_not_touch_offline_games(self):
        first, second = self.base/'offline-drive/GameA', self.base/'offline-drive/GameB'
        write_library(self.source, {'A': str(first)})
        write_library(self.target, {'B': str(second)}, ['Hidden'])
        before = (self.target/'.scout-library.json').read_bytes()
        source_before = (self.source/'.scout-library.json').read_bytes()
        self.assertEqual(import_library_index(self.source/'.scout-library.json', self.target), 1)
        self.assertEqual(read_library(self.target)['games'], {'A': str(first), 'B': str(second)})
        self.assertEqual(read_library(self.target)['removed_legacy'], ['Hidden'])
        backups = list(self.target.glob('.scout-library.before-import-*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertEqual((self.source/'.scout-library.json').read_bytes(), source_before)
        self.assertFalse(first.parent.exists())
        self.assertEqual(import_library_index(self.source/'.scout-library.json', self.target), 0)
        self.assertEqual(len(list(self.target.glob('.scout-library.before-import-*.json'))), 1)

    def test_conflicts_and_corruption_preserve_destination(self):
        write_library(self.source, {'Same': str(self.base/'different')})
        write_library(self.target, {'Same': str(self.base/'original')})
        before = (self.target/'.scout-library.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Conflicting game names'):
            import_library_index(self.source/'.scout-library.json', self.target)
        self.assertEqual((self.target/'.scout-library.json').read_bytes(), before)
        (self.source/'.scout-library.json').write_text('{broken')
        with self.assertRaisesRegex(ValueError, 'Cannot read'):
            import_library_index(self.source/'.scout-library.json', self.target)
        self.assertEqual((self.target/'.scout-library.json').read_bytes(), before)

    def test_optimistic_check_prevents_overwriting_concurrent_edits(self):
        write_library(self.target, {'A': str(self.base/'A')})
        digest = index_digest(self.target)
        write_library(self.target, {'B': str(self.base/'B')})
        with self.assertRaisesRegex(ValueError, 'changed'):
            write_library(self.target, {'A': str(self.base/'A')}, expected_digest=digest, backup=True)
        self.assertEqual(set(read_library(self.target)['games']), {'B'})
        self.assertEqual(list(self.target.glob('.scout-library.before-import-*.json')), [])

    def test_import_into_empty_user_data_and_reject_self_import(self):
        write_library(self.source, {'A': str(self.base/'A')})
        self.assertEqual(import_library_index(self.source/'.scout-library.json', self.target), 1)
        with self.assertRaisesRegex(ValueError, 'already open'):
            import_library_index(self.target/'.scout-library.json', self.target)
        self.assertFalse((self.base/'A').exists())

    def test_writer_lock_is_released_after_failure(self):
        write_library(self.target, {})
        with self.assertRaises(ValueError):
            write_library(self.target, {}, expected_digest='stale')
        write_library(self.target, {'A': str(self.base/'A')})
        self.assertEqual(set(read_library(self.target)['games']), {'A'})

    def test_named_backup_can_be_imported_without_renaming(self):
        write_library(self.source, {'A': str(self.base/'A')})
        backup = self.base/'my-backup.json'
        backup.write_bytes((self.source/'.scout-library.json').read_bytes())
        self.assertEqual(import_library_index(backup, self.target), 1)
        self.assertEqual(set(read_library(self.target)['games']), {'A'})

    def test_folder_import_cannot_erase_a_concurrent_registry_import(self):
        from scout.desktop import library
        folder = self.base/'NewGame'
        folder.mkdir()
        write_library(self.target, {'A': str(self.base/'A')})
        original = library.state_directory
        def concurrent_import(game):
            write_library(self.target, {'A': str(self.base/'A'), 'B': str(self.base/'B')})
            return original(game)
        with patch.object(library, 'state_directory', side_effect=concurrent_import):
            with self.assertRaisesRegex(ValueError, 'changed'):
                library.import_folder(self.target, folder)
        self.assertEqual(set(read_library(self.target)['games']), {'A', 'B'})

    def test_match_removal_cannot_erase_a_concurrent_registry_import(self):
        from scout.desktop import library
        write_library(self.target, {'A': str(self.base/'A')})
        original = library._library_index
        count = 0
        def concurrent_read(root):
            nonlocal count
            count += 1
            snapshot = original(root)
            if count == 1:
                write_library(root, dict(snapshot['games'], B=str(self.base/'B')))
            return snapshot
        with patch.object(library, '_library_index', side_effect=concurrent_read):
            with self.assertRaisesRegex(ValueError, 'changed'):
                library.remove_match(self.target, 'A')
        self.assertEqual(set(read_library(self.target)['games']), {'A', 'B'})
