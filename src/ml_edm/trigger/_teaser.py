import copy
import random
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from scipy.stats import hmean
from joblib import Parallel, delayed
from warnings import warn

from ._base import BaseTriggerModel

class TEASER(BaseTriggerModel):

    def __init__(self,
                 timestamps,
                 objective='hmean',
                 n_jobs=1):
        
        super().__init__(timestamps)

        ######Constant attributes#######
        self.require_past_probas = True
        ################################

        self.objective = objective
        self.n_jobs = n_jobs

    def _generate_features(self, probas):

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        #preds = probas.argmax(axis=-1)

        features = np.concatenate(
            (probas, diff[:, None]), axis=-1
        )

        return features 
    
    def _fit_master_clf(self, probas, masks_pos_probas):

        oc_features = self._generate_features(probas)

        oc_clf = OneClassSVM(kernel='rbf', nu=.05, tol=1e-4)
        gamma_grid = (
            {"gamma": [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]}
        )
        if len(oc_features[masks_pos_probas]) > 2:
            gs = GridSearchCV(
                estimator=oc_clf,
                param_grid=gamma_grid,
                scoring='accuracy',
                cv=min(len(oc_features[masks_pos_probas]), 10),
                n_jobs=1
            )
            # train the set of master classifiers only
            # on positives samples 
            gs.fit(oc_features[masks_pos_probas], 
               np.ones(len(oc_features[masks_pos_probas]))
            )
            clf = gs.best_estimator_
        else:
            oc_clf.set_params(**{"gamma": random.sample(gamma_grid['gamma'], 1)[0]})
            clf = oc_clf.fit(oc_features[masks_pos_probas])
            warn("Not enough positives samples to learn from, "
                 "selecting gamma value randomly")

        return clf, oc_features
    
    def _fit(self, X_probas, y):
        
        self.n_classes = len(self.classes_)

        # get predictions for each timestamp
        X_pred = X_probas.argmax(axis=-1)
        masks_pos_probas = np.array([x==y for x in X_pred.T])

        res = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._fit_master_clf)(X_probas[:,j,:], masks_pos_probas[j]) 
             for j in range(len(self.timestamps)))
        self.master_clfs, X_oc = zip(*res)

        # search best threshold v
        best_obj = np.inf
        for v in np.arange(1, 6):
            final_pred = np.ones(len(X_probas)) * (-1)
            final_t_star = copy.deepcopy(final_pred)
            consecutive_pred = np.zeros(len(X_probas))

            for j, t in enumerate(self.timestamps):
                accept = (self.master_clfs[j].predict(X_oc[j]) == 1)
                if j==0:
                    consecutive_pred[accept] += 1
                else:
                    prev_pred = X_pred[:,j-1]
                    current_pred = X_pred[:,j]
                    consecutive_pred[accept & (prev_pred == current_pred)] += 1
                    # reset the count if streak is broken 
                    consecutive_pred[accept & (prev_pred != current_pred)] = 0
                
                threshold_mask = (consecutive_pred == v)
                final_pred[threshold_mask & (final_pred < 0)] = X_pred[:,j][threshold_mask & (final_pred < 0)]
                final_t_star[threshold_mask & (final_t_star < 0)] = t

                if -1 not in final_pred:
                    break
                
                # if final timestep is reached, samples that hasn't been 
                # triggered yet are trigger 
                if t == self.timestamps[-1]:
                    final_pred = np.where(final_pred == -1, current_pred, final_pred)
                    final_t_star = np.where(final_t_star == -1, t, final_t_star)
            
            if self.objective == 'hmean':
                acc = (final_pred == y).mean()
                earliness_gain = 1 - (np.mean(final_t_star) / self.max_length)
                # minimize inverse of highest hmean
                if best_obj > -hmean((acc, earliness_gain)): # add weights here ?
                    best_obj = -hmean((acc, earliness_gain))
                    self.best_v = v
            else:
                t_idx = [
                    np.where(self.timestamps == final_t_star[i])[0][0] 
                    for i in range(len(final_t_star))
                ]
                avg_cost = np.mean(
                    [self.cost_matrices[int(t_idx[i])][int(p)][y[i]]
                     for i, p in enumerate(final_pred)]
                )
                if best_obj > avg_cost:
                    best_obj = avg_cost
                    self.best_v = v

        return self
    
    def _predict(self, X_probas, X_timestamps):

        triggers = []
        for i, probas in enumerate(X_probas):
            if np.where(self.timestamps == X_timestamps[i])[0][0]+1 < self.best_v:
                triggers.append(False)
            else:
                probas = np.array(probas).reshape((-1, self.n_classes))
                pred = probas.argmax(axis=-1)
                oc_features = self._generate_features(probas)

                accept = np.array(
                    [(self.master_clfs[l].predict(oc_features[l:l+1]) == 1)
                     for l in range(len(probas))]
                )
                
                trigger = False
                for i in range(len(pred)-self.best_v+1):
                    window_pred = pred[i:i+self.best_v]
                    window_accept = accept[i:i+self.best_v]

                    if (np.all(window_pred == window_pred[0])) & \
                        (np.sum(window_accept) == self.best_v):
                        trigger = True
                        triggers.append(trigger)
                        break
                
                if not trigger:
                    triggers.append(False)

        return np.array(triggers)