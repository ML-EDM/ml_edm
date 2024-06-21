import copy
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

from ._base import BaseTriggerModel

class ECDIRE(BaseTriggerModel):

    def __init__(self,
                 chronological_classifiers,
                 threshold_acc=1.,
                 cross_validation=False,
                 n_jobs=1):

        self.timestamps = copy.deepcopy(chronological_classifiers.timestamps)
        super().__init__(self.timestamps)

        ######Constant attributes#######
        self.alter_classifiers = True
        ################################

        self.chronological_classifiers = copy.deepcopy(chronological_classifiers)
        self.chronological_classifiers.prev_models_input_lengths = self.timestamps

        self.threshold_acc = threshold_acc
        self.cross_validation = cross_validation
        self.n_jobs = n_jobs

    def _fit_cv(self, X, y, train_idx, test_idx):

        classifiers_cv, acc_cv, probas_cv = [], [], []

        clfs = copy.deepcopy(self.chronological_classifiers)
        clfs.classifiers = None
        clfs.fit(X[train_idx], y[train_idx])
        classifiers_cv.extend(clfs.classifiers)

        preds = [clfs.classifiers[j].predict(X[test_idx, :t])
                 for j, t in enumerate(clfs.timestamps)]
        probs = [clfs.classifiers[j].predict_proba(X[test_idx, :t]) 
                 for j, t in enumerate(clfs.timestamps)]
        
        acc_cv, probas_cv = self._get_metrics(probs, preds, y[test_idx])

        return acc_cv, probas_cv

    def _get_metrics(self, probas, predictions, y_true):

        accuracies, probs = [], []

        matrices = [confusion_matrix(y_true, pred) for pred in predictions]
        # get intra-class accuracies 
        accuracies.append(
            [matrix.diagonal()/matrix.sum(axis=1) for matrix in matrices]
        )

        correct_mask = [(pred == y_true) for pred in predictions]
        correct_probs = [p[correct_mask[j]] for j, p in enumerate(probas)]
        # get probabilities for correct predicted samples 
        probs.append(
            [np.concatenate((probs, y_true[correct_mask[j], None]), axis=-1)
             for j, probs in enumerate(correct_probs)]
        )

        return accuracies, probs
    
    def _get_timeline(self, mean_acc):

        timeline = []
        for c, acc_class in enumerate(mean_acc.T):
            acc_threshold = acc_class[-1] * self.threshold_acc
            condition = np.where((acc_class < acc_threshold))[0]
            # if all timestamps are, at least, as great as the threshold
            if len(condition) == 0: 
                timestamp_idx = 0
            else: # if at least one timestamp is lower than thrseshold
                timestamp_idx = condition[-1] + 1 
            timestamp = self.timestamps[timestamp_idx]
            safe_timestamps = [t[0] for t in timeline] if len(timeline) > 0 else []

            if timestamp in safe_timestamps:
                timeline[safe_timestamps.index(timestamp)][1].add(c)
            else:
                timeline.append((timestamp, {c}))

        return sorted(timeline)
    
    def _get_reliability(self, probas):

        thresholds = []
        for j in range(len(self.timestamps)):
            probas_t = np.vstack([clf[j] for clf in probas])
            
            class_thresholds = []
            for c in range(probas_t.shape[-1] - 1):
                probas_c = probas_t[(probas_t[:,-1] == c)][:, :-1]
                
                if len(probas_c) == 0:
                    probas_diff = np.array([0])
                else:
                    probas_diff = np.array([np.max(x)-x for x in probas_c])
                    probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
                    probas_diff = np.min(probas_diff, axis=-1)

                class_thresholds.append((np.min(probas_diff)))
            
            thresholds.append(class_thresholds)
        
        return np.array(thresholds)
        
    def _fit(self, X_probas, y):

        if self.cross_validation:
            rskf = RepeatedStratifiedKFold(random_state=42)
            results_cv = Parallel(n_jobs=self.n_jobs) \
                (delayed(self._fit_cv)(self.ts, y, train_idx, test_idx) for train_idx, test_idx in rskf.split(self.ts, y)) 
        
            acc_cv, probs_cv = list(map(list, zip(*results_cv)))
            probs_cv = [p[0] for p in probs_cv]
        else:
            acc_cv, probs_cv = self._get_metrics(
                np.swapaxes(X_probas, 0, 1), X_probas.argmax(-1).T, y
            )

        mean_acc = np.array(acc_cv).mean(axis=0).squeeze()
        self.timeline = self._get_timeline(mean_acc)
        self.reliability = self._get_reliability(probs_cv)

        max_t = max(self.timeline)[0]
        if max_t != self.timestamps[-1]:
            idx = np.where(self.timestamps == max_t)[0][0] + 1
            self.timeline.extend(
                ((t, set(y)) for t in self.timestamps[idx:])
            )
        else:
            self.timeline[-1] = (max_t, set(y))
        
        clf_idx = [np.where(self.timestamps==t)[0][0]
                   for t in list(zip(*self.timeline))[0]]
        
        self.reliability = self.reliability[clf_idx]
        self.timestamps = self.timestamps[clf_idx]
        self.chronological_classifiers.timestamps = self.timestamps
        new_classifiers = [
            self.chronological_classifiers.classifiers[j] 
            for j in clf_idx
        ]
        self.chronological_classifiers.classifiers = new_classifiers

        if len(self.chronological_classifiers.extractors) > 0:
            self.chronological_classifiers.extractors = [
                self.chronological_classifiers.extractors[j]
                for j in clf_idx
            ]

        return self
    
    def _predict(self, X_probas, X_timestamps):

        triggers = []
        for i, probas in enumerate(X_probas):
            trigger  = False
            if X_timestamps[i] == 0:
                triggers.append(trigger)
            else:
                pred = probas.argmax(axis=-1)
                safe_class = [c for t, c in self.timeline
                              if X_timestamps[i] == t]
                if pred in safe_class[0]:
                    probas_diff = np.array([np.max(probas)-x for x in probas])
                    probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
                    probas_diff = np.min(probas_diff)

                    t_idx = np.where(self.timestamps == X_timestamps[i])
                    if probas_diff >= self.reliability[t_idx[0][0], pred]:
                        trigger = True
                triggers.append(trigger)

        return np.array(triggers)