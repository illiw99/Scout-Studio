import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scout.config import Config
from scout import model


def script(filename='benchmark-classifiers.py'):
    path = Path(__file__).resolve().parents[1] / 'scripts' / filename
    spec = importlib.util.spec_from_file_location('classifier_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClassifierBenchmarkTests(unittest.TestCase):
    def test_duration_uses_full_source_timeline_for_older_confirmation_manifest(self):
        module = script()
        self.assertEqual(module.duration(dict(clips=[dict(start=0., duration=3.), dict(start=3., duration=5.)])), 8.)
    def test_safe_context_and_lost_action_get_different_diagnostics(self):
        module = script('report-classifier-benchmark.py')
        early = module.boundary_diagnostics([([[2, 6]], [[0, 8]])])
        cut = module.boundary_diagnostics([([[2, 6]], [[4, 5]])])
        self.assertEqual(early['covers_play_within_tolerance_recall'], 1.)
        self.assertEqual(early['both_boundaries_accurate_recall'], 0.)
        self.assertEqual(early['mean_lost_action_seconds'], 0.)
        self.assertEqual(early['mean_extra_context_seconds'], 4.)
        self.assertEqual(cut['covers_play_within_tolerance_recall'], 0.)
        self.assertEqual(cut['mean_lost_action_seconds'], 3.)
        self.assertEqual(cut['matched_plays_cut_by_more_than_tolerance'], 1)

    def test_boundaries_and_missed_plays_are_separate_from_overlap_hits(self):
        module = script()
        case = dict(t=np.arange(30.), record=dict(name='game', duration=30., marks=[[2, 6], [15, 20]]))
        with patch.object(module.model, 'postprocess', return_value=[(0., 8.)]):
            result = module.score([case], [np.zeros(30)], Config(sample_fps=1))
        self.assertEqual(result['event']['recall'], .5)
        self.assertEqual(result['event']['precision'], 1.)
        self.assertEqual(result['boundary_recall']['1.0'], 0.)
        self.assertEqual(result['boundary_recall']['2.0'], .5)
        self.assertEqual(result['start_p90'], 2.)
        self.assertEqual(result['end_p90'], 2.)
        self.assertEqual(result['event']['extra_seconds_per_detected_play'], 4.)

    def test_duplicate_games_and_shared_training_holdout_days_rejected(self):
        module = script()
        records = [dict(name=name, group=group, partition=part, marks=[[1, 3]]) for name, group, part in
                   [('a', 'one', 'development'), ('b', 'two', 'development'), ('c', 'three', 'holdout')]]
        self.assertEqual(module.validate_records(records), ['one', 'two'])
        with self.assertRaises(ValueError):
            module.validate_records(records + [records[0]])
        records[-1]['group'] = 'one'
        with self.assertRaises(ValueError):
            module.validate_records(records)

    def test_holdout_is_loaded_after_frozen_selection_and_never_fitted(self):
        module = script()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, output = root / 'input', root / 'output'
            (study / 'person').mkdir(parents=True)
            records = []
            for index, (name, group, partition) in enumerate([
                ('a', 'one', 'development'), ('b', 'two', 'development'), ('c', 'three', 'holdout')]):
                records.append(dict(name=name, group=group, partition=partition, marks=[[2, 5]], duration=10.))
                np.savez(study / 'person' / (name + '.npz'), t=np.arange(10.),
                         X=np.full((10, 1), index + 1.), names=np.asarray(['feature']))
            cfg = Config(sample_fps=1, model_offsets=module.OFFSETS, feature_pipeline='dynamic-field-v1')
            (study / 'manifest.json').write_text(json.dumps(records))
            (study / 'baseline.json').write_text(json.dumps(dict(configuration=asdict(cfg),
                classifier=dict(names=model.window_names(['feature'], module.OFFSETS)))))
            fitting, loaded = [], []
            class Classifier:
                def fit(self, X, y):
                    fitting.append(set(X[:, 0]))
                    return self
                def predict_proba(self, X):
                    return np.tile([.5, .5], (len(X), 1))
            original = module.load_case
            def load(*args):
                if args[1]['partition'] != 'development':
                    self.assertTrue((output / 'selection.json').is_file())
                loaded.append(args[1]['name'])
                return original(*args)
            def tune(cases, probabilities, config, target):
                return dict(parameters={}, **module.score(cases, probabilities, config)), 1
            args = SimpleNamespace(study=study, out=output, threads=1, seed=17,
                                   variants=['hgb-reference'], minimum_recall=.95)
            with patch.object(module, 'make_classifier', return_value=Classifier()), \
                 patch.object(module, 'load_case', side_effect=load), patch.object(module, 'tune', side_effect=tune):
                self.assertEqual(module.run(args), 0)
            self.assertEqual(fitting, [{2.}, {1.}, {1., 2.}])
            self.assertEqual(loaded, ['a', 'b', 'c'])
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(report['inputs_unchanged'])
            with self.assertRaises(ValueError):
                module.run(args)
            args.reuse_selection = output / 'selection.json'
            args.out = root / 'resumed'
            fitting.clear()
            with patch.object(module, 'make_classifier', return_value=Classifier()), \
                 patch.object(module, 'tune', side_effect=AssertionError('Resume must not retune')):
                self.assertEqual(module.run(args), 0)
            self.assertEqual(fitting, [{1., 2.}])


if __name__ == '__main__':
    unittest.main()
