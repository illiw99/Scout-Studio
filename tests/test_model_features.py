"""Play ratios/context and model recipe compatibility, without real footage."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scout import features, model, model_features
from scout.config import Config
from scout.groundtruth import Marked


class DerivedPlayFeaturesTests(unittest.TestCase):
    def observations(self, n=24):
        X = np.zeros((n, len(features.FEATURE_NAMES)), np.float32)
        for name, value in (('n_field', 10.), ('hull', .2), ('x_span', .4), ('coherence', .25)):
            X[:, features.FEATURE_NAMES.index(name)] = value
        return X

    def test_ratios_and_burst_context_use_observed_edges_and_do_not_mutate_input(self):
        X = self.observations()
        X[10:14, features.FEATURE_NAMES.index('n_active')] = 10
        before = X.copy()
        expanded, names = model_features.expand(X, features.FEATURE_NAMES, 1., model_features.CONTEXT)
        column = lambda name: expanded[:, names.index(name)]
        np.testing.assert_array_equal(X, before)
        self.assertEqual(column('active_fraction')[10], 1.)
        self.assertEqual(column('activity_rise')[10], 1.)
        self.assertEqual(column('activity_drop')[13], 1.)
        self.assertEqual(column('activity_rise')[0], 0.)
        self.assertEqual(column('activity_drop')[-1], 0.)
        self.assertEqual(column('motion_disorder')[10], .75)
        self.assertTrue(np.isfinite(expanded).all())

    def test_zero_players_missing_observation_and_constant_time_scaling(self):
        for fps in (1., 2.5, 5.):
            X = self.observations(round(12 * fps))
            X[:, features.FEATURE_NAMES.index('n_active')] = 5
            X[0, features.FEATURE_NAMES.index('n_field')] = 0
            X[1, features.FEATURE_NAMES.index('n_active')] = np.nan
            expanded, names = model_features.expand(X, features.FEATURE_NAMES, fps, model_features.CONTEXT)
            self.assertFalse(np.isinf(expanded).any())
            self.assertEqual(expanded[0, names.index('active_fraction')], 0.)
            self.assertEqual(expanded[-1, names.index('active_fraction')], .5)
            self.assertAlmostEqual(expanded[-1, names.index('activity_rise')], 0.)
            self.assertAlmostEqual(expanded[-1, names.index('activity_drop')], 0.)

    def test_recipe_and_offsets_are_explicit_and_bounded(self):
        for offsets in ((-1., 1.), (-1., 0., 0.), (-31., 0., 1.), (False, 1.)):
            with self.assertRaises(ValueError): model_features.validate_offsets(offsets)
        with self.assertRaises(ValueError):
            Config(model_feature_set=model_features.CONTEXT).validate_pipeline()
        with self.assertRaises(ValueError):
            model_features.expand(self.observations(), features.FEATURE_NAMES, 1., 'unknown')

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'training reference needed')
    def test_custom_context_pickle_and_numeric_export_predict_identically_and_reject_wrong_recipe(self):
        cfg = Config(feature_pipeline='dynamic-field-v1', model_feature_set=model_features.CONTEXT,
                     model_offsets=(-10., -6., -3., -1., 0., 1., 3., 6., 10.), sample_fps=2.5,
                     model_max_iter=3, model_min_leaf=3)
        X = self.observations(120)
        X[30:50, features.FEATURE_NAMES.index('n_active')] = 8
        f = features.Features(np.arange(len(X)) / cfg.sample_fps, X, features.FEATURE_NAMES)
        ds = model.build(cfg, [('fixture', f, [Marked(1, 12., 20.)])])
        clf = model.fit(cfg, ds)
        expected = model.predict(clf, f, cfg)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.json', '.pkl'):
                path = Path(directory) / ('head' + suffix)
                model.save(path, clf, ds.names, cfg)
                loaded, names = model.load(path, cfg)
                self.assertEqual(names, ds.names)
                np.testing.assert_allclose(model.predict(loaded, f, cfg), expected, rtol=0, atol=1e-12)
                wrong = copy.copy(cfg); wrong.model_feature_set = model_features.BASE
                with self.assertRaisesRegex(ValueError, 'feature set differs'): model.load(path, wrong)
                wrong = copy.copy(cfg); wrong.model_offsets = model_features.DEFAULT_OFFSETS
                with self.assertRaisesRegex(ValueError, 'temporal context'): model.load(path, wrong)
                if suffix == '.json':
                    document = json.loads(path.read_text())
                    self.assertEqual(document['format'], 'scout-hgb-v3')
                    document['names'][0], document['names'][1] = document['names'][1], document['names'][0]
                    path.write_text(json.dumps(document))
                    with self.assertRaisesRegex(ValueError, 'feature names'): model.load(path, cfg)
                else:
                    import pickle
                    with path.open('rb') as stream: metadata = pickle.load(stream)
                    self.assertEqual(metadata['format'], model.CONTEXT_PICKLE_FORMAT)
                    self.assertNotIn('clf', metadata)  # Old readers must fail safely.
