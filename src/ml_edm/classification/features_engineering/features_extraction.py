import numpy as np
from warnings import warn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection.feature_based import TSFresh
from aeon.transformations.collection.convolution_based import MiniRocket

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
            self.extractor = TSFresh(**self.kwargs)
        elif self.method == 'weasel2.0':
            self.min_length = 4
            self.extractor = WEASELTransformerV2(**self.kwargs)
        else:
            raise ValueError("Unknown features extraction method")

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)
            #X = self.scaler.fit_transform(X, y)
            self.extractor = make_pipeline(
                self.extractor,
                self.scaler
            )

        if X.shape[1] >= self.min_length:
            try:
                self.extractor = self.extractor.fit(np.expand_dims(X, 1), y).transform
            except AttributeError:
                self.extractor.fit_transform(np.expand_dims(X, 1), y)
                self.extractor = self.extractor.transform
        else:
            warn(f"Time series provided are too short, (length = {X.shape[1]}) for {self.method},"
                 " using timestamps as features")
            self.extractor = self._do_nothing

        return self
    
    def _do_nothing(self, x, y=None):
        return x.squeeze()
    
    def transform(self, X, y=None):
        #if self.scale:
        #    X = self.scaler.transform(X)
        return np.array(self.extractor(np.expand_dims(X, 1))).reshape(len(X),-1)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)