"""Automatic labels follow match time while keeping coaching annotations."""
import unittest
from scout.desktop.labels import label_for, time_label


class LabelTests(unittest.TestCase):
    def test_precise_match_clock_and_rounding(self):
        self.assertEqual(time_label(75.1234), 'Play 00:01:15.123')
        self.assertEqual(time_label(3599.9999), 'Play 01:00:00.000')
        self.assertEqual(time_label(3661.04), 'Play 01:01:01.040')

    def test_generated_labels_follow_retimed_boundaries(self):
        for text in ['', 'Play', 'Play 4', 'Play 00:01:15.123']:
            self.assertEqual(label_for(text, 80.), 'Play 00:01:20.000')

    def test_custom_annotations_are_kept(self):
        for text in ['Touchdown', 'Play action pass', 'Defense — blitz']:
            self.assertEqual(label_for(text, 80.), text)
