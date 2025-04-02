import numpy as np
from itertools import product, permutations
from scipy.stats import hmean
from joblib import Parallel, delayed

from ._base import BaseTriggerModel

class StoppingRule(BaseTriggerModel):

    """
    A myopic trigger model which triggers a decision whenever a stopping rule representing time and confidence passes a
    certain threshold. Trigger models aim to determine the best time to trigger a decision given evolving time series.
    Introduced in :
    
    [1] Early classification of time series by simultaneously optimizing the accuracy and earliness - U. Mori, A. Mendiburu,
    S. Dasgupta, J. A. Lozano - IEEE transactions on neural networks and learning systems
    """

    def __init__(self,
                 timestamps,
                 stopping_rule="SR1",
                 objective="avg_cost",
                 n_jobs=1):
        
        super().__init__(timestamps)

        self.stopping_rule = stopping_rule
        self.objective = objective
        self.n_jobs = n_jobs

    def _trigger(self, gammas, probas, t):

        if self.stopping_rule == "SR1":
            proba1 = np.max(probas) 
            proba2 = np.min(np.delete(np.abs(probas - proba1), probas.argmax())) 
            probas = np.array([proba1, proba2])

        score = gammas[:-1] @ probas + gammas[-1] * (t/self.max_length)

        if score > 0 or t==self.timestamps[-1]:
            return True

        return False 

    def _get_score(self, gammas, X_probas, y):

        costs = []
        for i in range(len(X_probas)):
            for j, t in enumerate(self.timestamps):
                trigger = self._trigger(gammas, X_probas[i][j], t)

                if trigger:
                    pred = np.argmax(X_probas[i][j])
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
        
        nb_gammas = 3 if self.stopping_rule == "SR1" else X_probas.shape[2]+1
        self.candidates_gammas = list(product(np.linspace(-1, 1, 10), repeat=nb_gammas))

        gamma_costs = Parallel(n_jobs=self.n_jobs) \
                    (delayed(self._get_score)(gammas, X_probas, y) for gammas in self.candidates_gammas)
        
        self.opt_gammas = self.candidates_gammas[np.argmin(gamma_costs)]

        return self

    def _predict(self, X_probas, X_timestamps):
        
        triggers = []
        for i, p in enumerate(X_probas):
            triggers.append(
                self._trigger(self.opt_gammas, p, X_timestamps[i])
            )

        return np.array(triggers)