import os

import numpy as np
from warnings import warn

from sklearn.preprocessing import StandardScaler
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection.tsfresh import TSFreshFeatureExtractor
from aeon.transformations.collection.rocket import MiniRocket
#from sktime.transformations.panel.rocket import MiniRocket

class Feature_extractor:

    def __init__(self, method, scale=True, kwargs={}):

        self.method = method 
        self.scale = scale
        self.kwargs = kwargs
        self.min_length = -1 

    def fit(self, X, y=None):

        if self.method == 'minirocket':
            self.min_length = 9
            self.extractor = MiniRocket(**self.kwargs)
        elif self.method == 'tsfresh':
            self.extractor = TSFreshFeatureExtractor(**self.kwargs)
        else:
            raise ValueError("Unknown features extraction method")

        if self.scale:
            self.scaler = StandardScaler(with_mean=False).fit(X)
            X = self.scaler.transform(X)

        if X.shape[1] >= self.min_length:
            self.extractor = self.extractor.fit(np.expand_dims(X, 1), y).transform
        else:
            warn(f"Time series provided are too short for {self.method},"
                 "no extraction performed")
            self.extractor = self._do_nothing

        return self
    
    def _do_nothing(self, x, y=None):
        return x.squeeze()
    
    def transform(self, X, y=None):
        if self.scale:
            X = self.scaler.transform(X)
        return np.array(self.extractor(np.expand_dims(X, 1), y))
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)