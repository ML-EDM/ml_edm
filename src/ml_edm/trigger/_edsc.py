import numpy as np
from joblib import Parallel, delayed

from ml_edm.trigger._base import BaseTriggerModel

class EDSC(BaseTriggerModel):

    def __init__(self,
                 min_length,
                 max_length,
                 threshold_learning='che',
                 prob_threshold=.95,
                 bound_threshold=3,
                 alpha=6,
                 min_coverage=1.,
                 n_jobs=1):
        
        super().__init__()

        ######Constant attributes#######
        self.require_classifiers = False
        ################################

        self.min_length = min_length
        self.max_length = max_length
        self.threshold_learning = threshold_learning
        self.prob_threshold = prob_threshold
        self.bound_threshold = bound_threshold
        self.alpha = alpha
        self.min_coverage = min_coverage

        self.n_jobs = n_jobs

    def _get_bmd(self, X, length):

        sub, bmd = [], []
        for ts in X:
            sub.append(
                [ts[i:length+i] for i in range(len(ts)-length)]
            )
        sub = np.array(sub)

        for i in range(len(X)):
            bmd_f = []
            for shapelet in sub[i]:
                bmd_f.append(
                    self._compute_dist(sub, shapelet)
                )
            bmd.append(bmd_f)

        return np.array(bmd).squeeze(), sub

    def _compute_dist(self, X, shapelet):

        bmd = []
        dists = [[np.linalg.norm(shapelet - candidate) 
                    for candidate in xx] for xx in X]
        bmd.append(
            np.array(dists).min(axis=-1)
        )

        return bmd
     
    def _kernel_density(self, X, x, bandwidth):

        cst = 1/(np.sqrt(2*np.pi) * bandwidth)
        dens = np.exp(-(X - x)**2 / (2*bandwidth)**2).mean()

        return cst * dens 
    
    def _kde_threshold(self, bmd_list, y, idx):
        
        target = int(y[idx])
        bmd_list = np.insert(bmd_list, idx, 0.0)
        #y = np.delete(y, idx)
        
        prior_target = (y == target).mean()

        bmd_target = bmd_list[(y == target)]
        bmd_non_target = bmd_list[(y != target)]
        bmd_sort = np.sort(bmd_list)

        h_target = 1.06 * np.std(bmd_target) * (len(bmd_target)**(-0.2))
        h_non_target = 1.06 * np.std(bmd_non_target) * (len(bmd_non_target)**(-0.2))

        dens_target = -self._kernel_density(bmd_target, bmd_sort[-1]/len(y), h_target)
        dens_non_target = -self._kernel_density(bmd_non_target, bmd_sort[-1]/len(y), h_non_target)

        proba = self._compute_proba(prior_target, dens_target, dens_non_target)
        
        threshold = -1
        if proba > self.prob_threshold:
            dens_target = [self._kernel_density(bmd_target, bmd_sort[i], h_target)
                           for i in range(len(bmd_sort))]
            dens_non_target = [self._kernel_density(bmd_non_target, bmd_sort[i], h_non_target)
                               for i in range(len(bmd_sort))]
            probas = np.array(
                [self._compute_proba(prior_target, dens_target[i], dens_non_target[i])
                 for i in range(len(bmd_sort))]
            )
            break_idx = np.where(probas >= self.prob_threshold)[0]
            for idx in range(len(bmd_sort)):
                if idx not in break_idx:
                    break_idx = idx
                    break 

            if break_idx > 1:
                val_candidates = np.linspace(bmd_sort[break_idx-1], bmd_sort[break_idx], 20)
                dens_target = [self._kernel_density(bmd_target, val_candidates[i], h_target)
                               for i in range(len(val_candidates))]
                dens_non_target = [self._kernel_density(bmd_non_target, val_candidates[i], h_non_target)
                                   for i in range(len(val_candidates))]
                probas = np.array(
                    [self._compute_proba(prior_target, dens_target[i], dens_non_target[i])
                     for i in range(len(val_candidates))]
                )
                threshold = val_candidates[
                    np.where(probas >= self.prob_threshold)[0][-1]
                ]

        return threshold, target
        
    def _compute_proba(self, prior_target, dens_target, dens_non_target):

        proba = (prior_target * dens_target) / \
            (((1-prior_target) * dens_non_target) + (prior_target * dens_target))
        
        return proba

    def _che_threshold(self, bmd_list, y, idx):

        target = int(y[idx])
        y = np.delete(y, idx)
        bmd_non_target = bmd_list[(y != target)]

        threshold = np.mean(bmd_non_target) - self.bound_threshold * np.var(bmd_non_target)

        return threshold, target
    
    def _get_utility(self, X, y, shapelet, bmd_list, serie_idx):
        
        y = np.delete(y, serie_idx)
        bmd_list_tmp = bmd_list

        eml_list = []
        for i, ts in enumerate(X):
            
            if i == serie_idx: # if considered shapelet comes from this serie
                bmd_list_tmp = np.insert(bmd_list, i, 0.0)
                continue

            if shapelet[1] > bmd_list_tmp[i]:
                dists = [np.linalg.norm(shapelet[0] - ts[i:len(shapelet[0])+i])
                        for i in range(len(ts) - len(shapelet[0]))]
                eml = np.where(np.array(dists) <= shapelet[1])[0]
            else:
                eml = np.array([])

            eml = eml[0] if len(eml) > 0 else np.inf
            eml_list.append(eml+len(shapelet[0]))

        class_mask = (y == shapelet[-1])
        wrecall = np.power(eml_list, (-1/self.alpha))[class_mask].mean()
        #wrecall = np.power(eml_list, (-1/self.alpha)).sum() / (~class_mask).sum()
        
        thresh_mask = (np.array(bmd_list) <= shapelet[1])
        match_mask = (y[thresh_mask] == shapelet[-1]) 
        precision = np.mean(match_mask) if len(match_mask) > 0 else 0

        utility = 2 * precision * wrecall / (precision + wrecall)
        coverage = thresh_mask & class_mask

        return np.nan_to_num(utility), np.insert(coverage, serie_idx, True)

    def _learn_shapelets(self, X, y, bmd, length):
        
        shapelets = []
        for i in range(len(X)):

            if length == 0:
                bmd_temp = bmd[i,:,length].T
            else:
                bmd_temp = bmd[i,:,length].T[:-length]

            for j, bmd_list in enumerate(bmd_temp):

                if self.threshold_learning == 'kde':
                    p = self._kde_threshold(bmd_list, y, i)
                elif self.threshold_learning == 'che':
                    p = self._che_threshold(bmd_list, y, i)
                    
                if p[0] > 0:
                    feature = (X[i][j:j+self.min_length+length], p[0], p[1])
                    # self.subsequences[length][i,j,:]
                    utility, coverage = self._get_utility(X, y, feature, bmd_list, i)
                    shapelets.append((feature, coverage, utility))

        return shapelets

    def _get_bmd_pair(self, arr1, arr2):

        dists = [np.full((len(arr1)-self.min_length+1, len(arr1)-self.min_length+1), np.nan, dtype=np.float16)
                 for _ in range(self.n_lengths)]

        for start1 in range(len(arr1) - self.min_length + 1):
            for start2 in range(len(arr2) - self.min_length + 1):

                d_squared = 0
                sub1 = arr1[start1 : start1+self.min_length]
                sub2 = arr2[start2 : start2+self.min_length]
                d_squared = np.linalg.norm(sub1 - sub2) ** 2
                dists[0][start1, start2] = np.sqrt(d_squared)
                
                offset1 = np.maximum(start1 - (len(arr1) - self.max_length), 0)
                offset2 = np.maximum(start2 - (len(arr1) - self.max_length), 0)

                lags1 = np.zeros((self.n_lengths-1,), dtype=int)
                if offset1 > 0:
                    lags1 = np.linspace(1, offset1, offset1, dtype=int)
                    lags1 = np.pad(lags1, (self.n_lengths-len(lags1)-1,0))
                
                lags2 = np.zeros((self.n_lengths-1,), dtype=int)
                if offset2 > 0:
                    lags2 = np.linspace(1, offset2, offset2, dtype=int)
                    lags2 = np.pad(lags2, (self.n_lengths-len(lags2)-1,0))

                for i, l in enumerate(range(self.n_lengths - np.maximum(offset1, offset2) - 1)):
                    idx1 = self.min_length + start1 + l
                    idx2 = self.min_length + start2 + l
                    d_squared += (arr1[idx1] - arr2[idx2])**2
                    dists[l+1][start1, start2] = np.sqrt(d_squared)

        return np.nanmin(dists, axis=2)
            
    def _fit(self, X, y):

        self.n_lengths = self.max_length - self.min_length + 1 
        bmd = np.zeros((len(X), len(X)-1, self.n_lengths, X.shape[1]-self.min_length+1), dtype=np.float16)

        print("Computing distances...")
        bmds = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_bmd_pair)(X[i], X[j]) 
             for i in range(len(X))
             for j in range(i+1, len(X)))
        
        for i in range(len(X)):
            idx = len(X) - i - 1
            if i == 0:
                bmd[i] = bmds[:int(idx)]
            elif i == len(X)-1:
                bmd[i] = prev
            else:
                bmd[i] = np.concatenate((prev, bmds[:int(idx)]), axis=0)
            prev = bmd[i][:i+1]
            bmds = bmds[int(idx):]

        # learn some distance threshold 
        print("Learning shapelets...")
        self.shapelets = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._learn_shapelets)(X, y, bmd, l) 
             for l in range(self.n_lengths))

        # release memory
        del bmd
        # flatten the list of list
        self.shapelets = sum(self.shapelets, [])

        # sort shapelets by utility
        self.shapelets = sorted(
            self.shapelets, key=lambda x: x[-1], reverse=True
        )
        self.features = []
        for shapelet in self.shapelets:

            if len(self.features) == 0:
                self.features.append(shapelet[0])
                current_cov = shapelet[1]
            elif (current_cov | shapelet[1]).sum() > current_cov.sum():
                self.features.append(shapelet[0])
                current_cov = current_cov | shapelet[1]
            
            if current_cov.mean() >= self.min_coverage:
                break

        return self
    
    def _predict(self, X, X_timestamps=None):
        
        all_preds, all_triggers, all_t_star = (np.full((len(X),), np.nan), 
                                               np.zeros((len(X),), dtype=bool), 
                                               np.full((len(X),), np.nan))
        min_L = min([len(fts[0]) for fts in self.features])

        for l in range(min_L, X.shape[1]):
            predictions, triggers, time_idx = [], [], []
            for ts in X[:,:l]:
                # test features by utility ?
                for j, fts in enumerate(self.features):
                    sub = fts[0]
                    if (len(sub) > len(ts)) and j != len(self.features)-1:
                        continue
                    elif (len(sub) > len(ts)) and j == len(self.features)-1:
                        triggers.append(False)
                        predictions.append(np.nan)
                        time_idx.append(X.shape[1])
                        break

                    dists = [np.linalg.norm(sub - ts[i:i+len(sub)])
                            for i in range(len(ts)-len(sub))]
                    
                    trigger = (np.array(dists) <= fts[1])
                    if trigger.any():
                        triggers.append(True)
                        predictions.append(fts[-1])
                        time_idx.append(np.where(trigger)[0][0]+len(fts[0]))
                        break

                    if j == len(self.features)-1:
                        triggers.append(False)
                        predictions.append(np.nan)
                        time_idx.append(X.shape[1])

            past_trigger = ~np.isnan(all_preds)
            if len(past_trigger) > 0:
                np.array(triggers)[past_trigger] = False

            all_preds[triggers] = np.array(predictions)[triggers]
            all_t_star[triggers] = np.array(time_idx)[triggers]
            all_triggers[triggers] = True

            if all_triggers.mean() == 1.:
                break

        return all_preds, all_triggers, np.nan_to_num(all_t_star, nan=X.shape[1])