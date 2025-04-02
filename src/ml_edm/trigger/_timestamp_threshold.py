import numpy as np
from joblib import Parallel, delayed

from ._base import BaseTriggerModel

class TimestampThreshold(BaseTriggerModel):
    """
    Simple trigger model that find best constant timestamp to trigger.
    Output classifiers predictions as soon as this timestamp is reached.

    """
    def __init__(self, 
                 timestamps, 
                 manual_threshold=None,
                 n_jobs=1):

        super().__init__(timestamps)

        self.manual_threshold = manual_threshold
        self.n_jobs = n_jobs

        if manual_threshold is not None:
            self.opt_threshold = manual_threshold

    def _get_score(self, threshold, X_probas, y):
        
        costs = []
        for i, probas in enumerate(X_probas):
            for j in range(len(self.timestamps)):
                trigger = (self.timestamps[j] >= threshold)

                if trigger or j==len(self.timestamps)-1:
                    pred = np.argmax(probas[j])
                    c = self.cost_matrices[j][pred][y[i]]
                    costs.append(c)
                    break
        
        agg_cost = np.mean(costs)

        return agg_cost
    
    def _fit(self, X_probas, y):
        
        if hasattr(self, 'opt_threshold'):
            return self
        
        n_classes = len(self.classes_)
        candidate_threshold = self.timestamps

        costs = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_score)(threshold, X_probas, y) for threshold in candidate_threshold)
            
        self.opt_threshold = candidate_threshold[np.argmin(costs)]

        return self

    def _predict(self, X_probas, X_timestamp):

        triggers = []
        for i in X_timestamp:
            triggers.append(
                (i >= self.opt_threshold)
            )
        return np.array(triggers)