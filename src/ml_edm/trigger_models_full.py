import numpy as np
from joblib import Parallel, delayed

class EDSC:

    def __init__(self,
                 min_length,
                 max_length,
                 threshold_learning='kde',
                 prob_trheshold=.95,
                 bound_threshold=3,
                 alpha=3,
                 min_coverage=1.,
                 n_jobs=1):
        
        self.min_length = min_length
        self.max_length = max_length
        self.threshold_learning = threshold_learning
        self.prob_threshold = prob_trheshold
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
    
    def _kde_threshold(self, bmd_list, y):

        target = int(y[(bmd_list == 0)])
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

    def _che_threshold(self, bmd_list, y):

        target = int(y[(bmd_list == 0)])
        bmd_non_target = bmd_list[(y != target)]

        threshold = np.mean(bmd_non_target) - self.bound_threshold * np.std(bmd_non_target)

        return threshold, target
    
    def _get_utility(self, X, y, shapelet):

        bmd_list, eml_list = [], []
        for ts in X:
            dists = [np.linalg.norm(shapelet[0] - ts[i:len(shapelet[0])+i])
                     for i in range(len(ts) - len(shapelet[0]))]
            
            bmd_list.append(np.min(dists))
        
            eml = np.where(np.array(dists) <= shapelet[1])[0]
            eml = eml[0] if len(eml) > 0 else np.inf
            eml_list.append(eml + len(shapelet[0]))

        class_mask = (y == shapelet[-1])
        wrecall = np.power(eml_list, (-1/self.alpha))[class_mask].mean()
        
        thresh_mask = (np.array(bmd_list) <= shapelet[1])
        precision = (y[thresh_mask] == shapelet[-1]).mean()

        utility = 2 * precision * wrecall / (precision + wrecall)
        coverage = thresh_mask & (y == shapelet[-1])

        return utility, coverage

    def _learn_shapelets(self, X, y, bmd, length):
        
        shapelets = []
        for i in range(len(X)):
            for j, bmd_list in enumerate(bmd[length][i]):

                if self.threshold_learning == 'kde':
                    p = self._kde_threshold(bmd_list, y)
                elif self.threshold_learning == 'che':
                    p = self._che_threshold(bmd_list, y)
                    
                if p[0] > 0:
                    feature = (self.subsequences[length][i,j,:], p[0], p[1])
                    utility, coverage = self._get_utility(X, y, feature)
                    shapelets.append((feature, coverage, utility))

        return shapelets

    def fit(self, X, y):
        
        self.subsequences = []
        results = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_bmd)(X, l) 
             for l in range(self.min_length, self.max_length+1))
        
        bmd, self.subsequences = list(map(list, zip(*results)))

        # learn some distance threshold 
        self.shapelets = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._learn_shapelets)(X, y, bmd, l) 
             for l in range(len(bmd)))
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
    
    def predict(self, X):

        predictions, triggers = [], []
        for ts in X:
            # test features by utility ?
            for j, fts in enumerate(self.features):
                sub = fts[0]
                if (len(sub) > len(ts)) and j != len(self.features)-1:
                    continue
                elif (len(sub) > len(ts)) and j == len(self.features)-1:
                    triggers.append(False)
                    predictions.append(np.nan)
                    break

                dists = [np.linalg.norm(sub - ts[i:i+len(sub)])
                         for i in range(len(ts)-len(sub)+1)]
                
                trigger = (np.array(dists) <= fts[1])
                if True in trigger:
                    triggers.append(True)
                    predictions.append(fts[-1])
                    break

                if j == len(self.features)-1:
                    triggers.append(False)
                    predictions.append(np.nan)

        return np.array(predictions), np.array(triggers)
    