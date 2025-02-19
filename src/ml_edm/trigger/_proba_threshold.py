import numpy as np
from scipy.stats import hmean

from joblib import Parallel, delayed
from ._base import BaseTriggerModel


class ProbabilityThreshold(BaseTriggerModel):
    """
    Probability threshold trigger model

    Trigger model based on some confidence threshold. The likeliest 
    prediction has to be to above the threshod to accept the decision.

    Parameters
    ----------

    timestamps : 
    """

    def __init__(self,
                 timestamps,
                 objective="avg_cost", 
                 manual_threshold=None,
                 n_jobs=1):
        
        super().__init__(timestamps)

        self.objective = objective
        self.manual_threshold = manual_threshold
        self.n_jobs = n_jobs

        if manual_threshold is not None:
            self.opt_threshold = manual_threshold
    
    def _get_score(self, threshold, X_probas, y):
        
        costs = []
        for i, probas in enumerate(X_probas):
            for j in range(len(self.timestamps)):
                trigger = (np.max(probas[j]) >= threshold)

                if trigger or j==len(self.timestamps)-1:
                    pred = np.argmax(probas[j])
                    c = self.cost_matrices[j][pred][y[i]] if self.objective=="avg_cost" \
                            else (self.cost_matrices.missclf_cost[j][pred][y[i]], self.cost_matrices.delay_cost[j][pred][y[i]])
                    costs.append(c)
                    break

        if self.objective=="avg_cost":
            agg_cost = np.mean(costs)
        elif self.objective=="hmean":
            # normalize as hmean only takes rates as input 
            acc  = 1 - np.mean([c[0] for c in costs]) / np.max(self.cost_matrices.missclf_cost)
            earliness_gain = 1 - np.mean([c[1] for c in costs]) / np.max(self.cost_matrices.delay_cost)
            agg_cost = -hmean((acc, earliness_gain)) # minimize inverse 
        else:
            raise ValueError("Unknown objective function, should be one of ['avg_cost', 'hmean']")

        return agg_cost
        
    def _fit(self, X_probas, y):
        
        if hasattr(self, 'opt_threshold'):
            return self
        
        n_classes = len(self.classes_)
        candidate_threshold = np.linspace(1/n_classes, 1, 41)

        costs = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_score)(threshold, X_probas, y) for threshold in candidate_threshold)
            
        self.opt_threshold = candidate_threshold[np.argmin(costs)]

        return self 

    def _predict(self, X_probas, X_timestamp):

        triggers = []
        for p in X_probas:
            triggers.append(
                (np.max(p) >= self.opt_threshold)
            )

        return np.array(triggers)

