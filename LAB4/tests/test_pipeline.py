import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from main import TrainQuantileClipper,add_features,validation,clean_tokens,make_preprocessor,CATEGORICAL,NUMERIC

class PipelineTests(unittest.TestCase):
    def test_clip_train_only(self):
        clip=TrainQuantileClipper().fit(np.array([[1.],[2.],[3.],[4.]]))
        self.assertLess(float(clip.transform([[1000.]])[0,0]),5)
    def test_clipper_feature_names(self):
        clip=TrainQuantileClipper().fit(np.array([[1.,2.],[3.,4.]]))
        self.assertEqual(list(clip.get_feature_names_out(["age","hours"])), ["age","hours"])
    def test_features(self):
        x=pd.DataFrame({'capital-gain':[100],'capital-loss':[20],'hours-per-week':[40],'age':[36]})
        y=add_features(x)
        self.assertEqual(y['capital_net'].iloc[0],80)
        self.assertEqual(y['hours_per_age'].iloc[0],2)
    def test_missing(self):
        row={k:1 for k in NUMERIC}
        row.update({k:'?' for k in CATEGORICAL})
        frame=clean_tokens(pd.DataFrame([row]))
        self.assertTrue(pd.isna(frame['workclass'].iloc[0]))
    def test_preprocessor(self):
        self.assertEqual(len(make_preprocessor(True).transformers),2)
if __name__=='__main__':unittest.main()
