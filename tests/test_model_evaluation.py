import unittest
import numpy as np
from scout.detect import Detections
from scout.model_evaluation import subsample, interval_metrics, summarize


class EvaluationTests(unittest.TestCase):
    def test_subsampling_selects_boxes_and_preserves_global_clock(self):
        det = Detections(10+np.arange(6)/5, np.array([0,1,1,3,4,4,5]),
                         np.arange(20).reshape(5,4), np.arange(5)/10, (100,200))
        sampled = subsample(det,5,1)
        np.testing.assert_equal(sampled.times,[10,11])
        np.testing.assert_equal(sampled.offsets,[0,1,2])
        np.testing.assert_equal(sampled.boxes,det.boxes[[0,4]])
        with self.assertRaises(ValueError):subsample(det,5,3)
        with self.assertRaises(ValueError):subsample(det,5,10)

    def test_short_overlap_does_not_hide_a_missed_play(self):
        m=interval_metrics([[10,20]],[[0,11]],.5)
        self.assertEqual((m['hits'],m['missed'],m['false_positives']),(0,1,1))
        m=interval_metrics([[10,20]],[[9,21]],.5)
        self.assertEqual(m['hits'],1)
        self.assertEqual((m['start_errors'],m['end_errors']),([-1],[1]))
        self.assertEqual(summarize([m])['boundary_p90'],1)

    def test_balanced_training_handles_rare_positive_labels(self):
        from scout.config import Config
        from scout.model import make_classifier
        X=np.zeros((100,2));y=np.r_[np.zeros(90),np.ones(10)]
        normal=make_classifier(Config(model_max_iter=2)).fit(X,y)
        balanced=make_classifier(Config(model_max_iter=2,model_class_weight='balanced')).fit(X,y)
        self.assertAlmostEqual(normal.predict_proba([[0,0]])[0,1],.1)
        self.assertAlmostEqual(balanced.predict_proba([[0,0]])[0,1],.5)
        with self.assertRaises(ValueError):make_classifier(Config(model_class_weight='unknown'))

    def test_event_recall_and_boundary_quality_are_distinct(self):
        event=interval_metrics([[10,20]],[[0,11]],0)
        self.assertEqual(event['hits'],1)
        summary=summarize([event])
        self.assertEqual(summary['recall'],1)
        self.assertEqual(summary['start_mae'],10)
        self.assertEqual(summary['end_mae'],9)
        self.assertEqual(summary['both_boundaries_within_seconds']['1.0'],0)
        self.assertEqual(summary['extra_seconds_per_detected_play'],10)
        self.assertEqual(summary['truncated_seconds_per_detected_play'],9)
        self.assertEqual(interval_metrics([[10,20]],[[0,10]],0)['hits'],0)

    def test_matching_is_one_to_one_and_empty_cases_are_counted(self):
        m=interval_metrics([[0,10],[11,20]],[[0,20]],.3)
        self.assertEqual((m['hits'],m['missed'],m['false_positives']),(1,1,0))
        self.assertEqual(interval_metrics([[0,10],[9,19]],[[0,12],[0,8]],.1)['hits'],2)
        self.assertEqual(interval_metrics([],[[0,2]])['false_positives'],1)
        self.assertEqual(interval_metrics([[0,2]],[])['missed'],1)
        with self.assertRaises(ValueError):interval_metrics([[2,1]],[])
