import numpy as np
from scipy.stats import hmean
from collections.abc import Iterable
from joblib import Parallel, delayed

from ._base import BaseTriggerModel

class ECEC(BaseTriggerModel):

    def __init__(self,
                 timestamps,
                 objective="avg_cost",
                 n_jobs=1):
        
        super().__init__(timestamps)

        ######Constant attributes#######
        self.require_past_probas = True
        ################################

        self.objective = objective
        self.n_jobs = n_jobs
        
    def _get_ratio(self, preds, y):

        ratios = []
        classes = np.unique(y)

        for y1 in classes:
            for y2 in classes:
                nominator = len(preds[(preds == y2) & (y == y1)])
                denominator = len(preds[(preds == y2)])

                if denominator != 0:
                    ratios.append(nominator / denominator)              
                else:
                    ratios.append(0)

        return ratios

    def _get_fused_confidence(self, preds):

        if not isinstance(preds, Iterable):
            preds = [preds]
        n_classes = np.sqrt(self.ratios.shape[-1]).astype(int)
        confidences = []
        for j in range(len(self.timestamps)):

            predictions = [preds[0]] if j==0 else preds[:j+1] 

            res = 1
            for k, p in enumerate(predictions):  
                col_idx = n_classes * predictions[-1] + p
                res *= (1 - self.ratios[k, col_idx])
            
            confidences.append(1 - res)

        return confidences

    def _get_score(self, threshold, X_pred, y):

        costs = []
        for i in range(len(X_pred)):

            confidences = self._get_fused_confidence(X_pred[i]) # replace by all confidences, already computed ?
            for j, c in enumerate(confidences):
                if c >= threshold:
                    pred = X_pred[i][j]
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

        X_pred = X_probas.argmax(axis=-1)

        # compute all ratios for all classifiers
        self.ratios = []
        for j in range(len(self.timestamps)):
            self.ratios.append(
                self._get_ratio(X_pred[:,j], y)
            )
        self.ratios = np.array(self.ratios)
        
        # compute fused confidence for all classifiers 
        all_confidences = []
        for i in range(len(X_pred)):
            all_confidences.extend(
                self._get_fused_confidence(X_pred[i])
            )

        candidates = np.unique(all_confidences)
        if len(candidates) > 1: # if only one candidates, pass
            candidates = [
                (candidates[j] + candidates[j+1]) / 2 for j in range(len(candidates)-1)
            ]

        # find best threshold candidates 
        costs = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_score)(threshold, X_pred, y) for threshold in candidates)
        
        self.opt_threshold = candidates[np.argmin(costs)]

        return self 
    
    def _predict(self, X_probas, X_timestamps):
        
        triggers = []
        for probas in X_probas:
            preds = np.array(probas).argmax(axis=-1)
            confidence = self._get_fused_confidence(preds)[-1]

            if confidence >= self.opt_threshold:
                triggers.append(True)
            else:
                triggers.append(False)

        return np.array(triggers)