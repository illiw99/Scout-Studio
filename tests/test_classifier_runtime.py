"""Exported inference matches training probabilities and play boundaries."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scout import model
from scout.config import Config
from scout.classifier_runtime import Classifier, load


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'training extra needed for equivalence reference')
class ClassifierExportTests(unittest.TestCase):
    def test_probabilities_missing_values_and_final_intervals_match(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng=np.random.default_rng(4)
        X=rng.normal(size=(300,3)); X[::9,1]=np.nan
        y=(X[:,0]+np.nan_to_num(X[:,1])>.1).astype(int)
        clf=HistGradientBoostingClassifier(max_iter=8,max_depth=3,min_samples_leaf=5,random_state=0).fit(X,y)
        cfg=Config(model_exclude=(),model_min_duration=.2,model_smooth=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json'
            model.save(path,clf,['a','b','c'],cfg)
            runtime,names=model.load(path,cfg)
            self.assertEqual(names,['a','b','c'])
            actual=runtime.predict_proba(X)
            expected=clf.predict_proba(X)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            times=np.arange(len(X))/cfg.sample_fps
            self.assertEqual(model.intervals(cfg,times,actual[:,1]),model.intervals(cfg,times,expected[:,1]))
            broken=Config(sample_fps=2,model_exclude=())
            with self.assertRaises(SystemExit):model.load(path,broken)


class ClassifierValidationTests(unittest.TestCase):
    def document(self):
        return dict(format='scout-hgb-v1',names=['x'],baseline=0.,trees=[dict(
            feature=[0,0,0],threshold=[0.,0.,0.],left=[1,0,0],right=[2,0,0],
            value=[0.,-1.,1.],missing_left=[True,False,False],leaf=[False,True,True])])

    def test_tree_traversal_missing_and_threshold_boundary(self):
        clf=Classifier(self.document())
        result=clf.predict_proba([[0.],[-1.],[1.],[np.nan]])[:,1]
        self.assertLess(result[0],.5)
        self.assertEqual(result[0],result[1]);self.assertEqual(result[0],result[3])
        self.assertGreater(result[2],.5)

    def test_cycles_indices_flags_and_nonfinite_values_are_rejected(self):
        for field,value in (('left',[0,0,0]),('right',[9,0,0]),('feature',[3,0,0]),
                            ('leaf',[0,True,True]),('value',[0.,np.nan,1.])):
            with self.subTest(field=field):
                data=self.document();data['trees'][0][field]=value
                with self.assertRaises(ValueError):Classifier(data)

    def test_input_schema_and_inf_fail(self):
        clf=Classifier(self.document())
        for data in ([[1.,2.]],[[np.inf]],np.zeros(3)):
            with self.assertRaises(ValueError):clf.predict_proba(data)
