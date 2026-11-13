"""Numeric boundary scoring preserves the frozen research contract and timeline."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from scout.boundary_runtime import BoundaryRefiner, feature_names, FORMAT, OFFSETS


class BoundaryRuntimeTests(unittest.TestCase):
    def document(self):
        names=feature_names(['x'])
        head=dict(format='scout-hgb-v1',names=names,baseline=0.,trees=[dict(
            feature=[len(names)-1,0,0],threshold=[.2,0.,0.],left=[1,0,0],right=[2,0,0],
            value=[0.,-1.,1.],missing_left=[True,False,False],leaf=[False,True,True])])
        return dict(format=FORMAT,names=['x'],heads=[head,copy.deepcopy(head)])

    def test_window_selection_ties_and_source_limits(self):
        refiner=BoundaryRefiner(self.document(),['x'])
        t=np.arange(0,10,.4);X=np.sin(t)[:,None];prob=np.full(len(t),.8)
        result=refiner.refine(t,X,prob,[(1,4),(7,9.9)],10.)
        np.testing.assert_allclose(result,[(1.4,4.4),(7.4,9.9+OFFSETS[np.argmin(abs(OFFSETS))])],atol=1e-12)

    def test_short_timeline_and_empty_proposals_preserve_valid_intervals(self):
        refiner=BoundaryRefiner(self.document(),['x'])
        self.assertEqual(refiner.refine([0],[[0]],[.8],[(0,.5)],.5),[(0,.5)])
        self.assertEqual(refiner.refine([],np.empty((0,1)),[],[],1),[])

    def test_independent_edge_shifts_cannot_reorder_or_add_overlaps(self):
        refiner=BoundaryRefiner(self.document(),['x'])
        class Head:
            def __init__(self,direction):self.direction=direction
            def predict_proba(self,X):
                scores=.5+self.direction*X[:,-1]/20
                return np.column_stack([1-scores,scores])
        refiner.heads=[Head(-1),Head(1)]
        t=np.arange(0,40,.4)
        original=[(10.,14.),(16.,20.)]
        self.assertEqual(refiner.refine(t,t[:,None],np.full(len(t),.8),original,40.),original)

    def test_corrupt_schema_and_nested_executable_formats_fail(self):
        for change in ('names','format','nested'):
            document=self.document()
            if change=='names':document['heads'][0]['names']=['wrong']
            elif change=='format':document['format']='unknown'
            else:document['heads'][0]['boundary_refiner']={}
            with self.assertRaises(ValueError):BoundaryRefiner(document,['x'])

    def test_invalid_observations_are_rejected(self):
        refiner=BoundaryRefiner(self.document(),['x'])
        for t,X,p,spans,duration in [([1,0],[[0],[0]],[.5,.5],[(0,2)],3),
                ([0],[[np.inf]],[.5],[(0,2)],3),([0],[[0]],[np.nan],[(0,2)],3),
                ([0],[[0]],[.5],[(-1,2)],3),([0],[[0]],[.5],[(0,4)],3)]:
            with self.assertRaises(ValueError):refiner.refine(t,X,p,spans,duration)

    def test_native_cli_segmentation_applies_refiner_to_shared_features(self):
        from scout.cli import _plays_from_model
        from scout.config import Config
        from scout.features import Features
        cfg=Config(model_exclude=(),model_min_duration=.2,model_smooth=0)
        f=Features(np.arange(10.)/cfg.sample_fps,np.ones((10,1)),['x'])
        refiner=Mock();refiner.refine.return_value=[(1.,2.)]
        classifier=SimpleNamespace(predict_proba=lambda X:np.tile([.1,.9],(len(X),1)),boundary_refiner=refiner)
        plays=_plays_from_model(cfg,classifier,f,3.)
        self.assertEqual([(p.start,p.end) for p in plays],[(1.,2.)])
        self.assertEqual(refiner.refine.call_args.args[1].shape,(10,len(cfg.model_offsets)))


if __name__=='__main__':unittest.main()
