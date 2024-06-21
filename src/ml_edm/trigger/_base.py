from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from warnings import warn

from ml_edm.cost_matrices import CostMatrices
from ml_edm.utils import *


class BaseTriggerModel(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, timestamps=None):

        self.require_past_probas = False
        self.require_classifiers = True
        self.alter_classifiers = False

        self.timestamps = timestamps
    

    def fit(self, X, X_probas, y, cost_matrices=None):
        
        X, y = check_X_y(X, y)

        if self.require_classifiers:
            X_probas = check_X_past_probas(X_probas)

            if len(y) != len(X_probas):
                raise ValueError("y should be an array of classes of size N with N the number of examples in X_probas.")

        self.max_length = X.shape[1]
        self.ts = X

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.class_prior = np.array([np.sum(y == class_) / len(y) 
                                     for class_ in self.classes_])

        self.cost_matrices = cost_matrices
        if self.cost_matrices is None:
            self.cost_matrices = CostMatrices(
                timestamps=self.timestamps, 
                n_classes=len(self.classes_), 
                alpha=0.5
            )
            warn("No cost matrices defined, using alpha = 0.5 by default")

        if self.timestamps is not None:
            self.timestamps = check_timestamps(self.timestamps)
            self.timestamps = np.sort(self.timestamps)
            if self.timestamps.shape[0] != len(self.cost_matrices):
                raise ValueError("Argument 'cost_matrices' should have as many matrices "
                                "as there are values in argument 'timestamps'.")
        else:
            self.timestamps = cost_matrices.timestamps

        if self.require_classifiers:
            self._fit(X_probas, y) 
        else:
            self._fit(X, y)

        return self

    def predict(self, X, X_probas, cost_matrices=None):
        
        ## get timepoints from series in X 
        ## closer to the self.timepoits defined
        if self.require_classifiers:
            if self.require_past_probas:
                X_probas = check_X_past_probas(X_probas)
            else:
                X_probas = check_X_probas(X_probas)
        else:
            X_probas = X

        # warning cost matrices test not the same as in self.cost_matrices !
        if cost_matrices is not None:
            if not isinstance(cost_matrices, CostMatrices):
                raise ValueError("cost_matrices should an object of CostMatrices class")
            
            if not np.array_equal(self.cost_matrices.values, cost_matrices.values):
                self.cost_matrices = cost_matrices
                warn("Different cost matrices between training and predicting"
                     "Performances may be hurted. ")

        X_timestamps = []
        for ts in X:
            diff = self.timestamps - len(ts)
            if 0 not in diff:
                truncated = True
                if (diff > 0).all(): # if ts smaller than all considered lengths
                    X_timestamps.append(0)
                else:
                    # replace bigger length by -inf and get smallest diff
                    time_idx = np.where(diff < 0, diff, -np.inf).argmax()
                    X_timestamps.append(self.timestamps[time_idx])
            else:
                # replace all non matching lengths by -inf and get index
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
                X_timestamps.append(self.timestamps[time_idx])

        return self._predict(X_probas, X_timestamps)
    
    @abstractmethod
    def _fit(self, X, y):
        """Fit the trigger model according to probas 
        outputed by classifiers (or not)

        """

    @abstractmethod
    def _predict(self, X, X_timestamps):
        """Predict whether or not time to make
        decision is safe or not 

        """
