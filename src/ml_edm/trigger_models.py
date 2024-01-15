import copy
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from abc import ABC, abstractmethod
from warnings import warn
from itertools import permutations
from collections.abc import Iterable
from joblib import Parallel, delayed
from scipy.stats import hmean

from sklearn.svm import OneClassSVM
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from utils import check_X_probas

np.seterr(divide='ignore', invalid='ignore')

def gini(probas):
    return 1 - np.sum(np.square(probas))

KNOWN_AGGREGATIONS = {"max": np.max, "gini": gini}

class TriggerModel(ABC):

    def __init__(self):
        self.require_past_probas = False
        self.require_classifiers = True
        self.alter_classifiers = False

    @abstractmethod
    def fit(self, X, y):
        """Fit the trigger model according to probas 
        outputed by classifiers (or not)

        """
    
    @abstractmethod
    def predict(self, X):
        """Predict whether or not time to make
        decision is safe or not 

        """


class ProbabilityThreshold(TriggerModel):

    def __init__(self,
                cost_matrices,
                models_input_lengths,
                objective="avg_cost",
                manual_threshold=None,
                n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.objective = objective

        if manual_threshold is not None:
            self.opt_threshold = manual_threshold
        
        self.n_jobs = n_jobs
    
    def _get_score(self, threshold, X_probas, y):
        
        costs = []
        for i, probas in enumerate(X_probas):
            for j in range(len(self.models_input_lengths)):
                trigger = (np.max(probas[j]) >= threshold)

                if trigger or j==len(self.models_input_lengths)-1:
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
            earliness = 1 - np.mean([c[1] for c in costs]) / np.max(self.cost_matrices.delay_cost)
            agg_cost = -hmean((acc, earliness)) # minimize inverse 
        else:
            raise ValueError("Unknown objective function, should be one of ['avg_cost', 'hmean']")

        return agg_cost
        
    def fit(self, X, X_probas, y, classes_=None):
        
        if hasattr(self, 'opt_threshold'):
            return self
        
        n_classes = len(np.unique(y))
        candidate_threshold = np.linspace(1/n_classes, 1, 41)

        costs = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_score)(threshold, X_probas, y) for threshold in candidate_threshold)
            
        self.opt_threshold = candidate_threshold[np.argmin(costs)]

        return self 

    def predict(self, X, X_probas):
        
        triggers = []
        for p in X_probas:
            triggers.append(
                (np.max(p) >= self.opt_threshold)
            )

        return np.array(triggers), None

        
class EconomyGamma(TriggerModel):
    """
    A highly performing non-myopic trigger model based on the economy architecture for early time series classification.
    Allows the anticipation of future decision costs based on a supervised grouping of time series and user-defined
    costs for delay and misclassification. Trigger models aim to determine the best time to trigger a decision given
    evolving time series. Can be used in junction with a Chronological Classifier object for early classification tasks.
    Introduced in paper [1].

    Parameters:
        cost_matrices: numpy.ndarray
            Array of matrices of shape (T, Y, Y) where T is the number of considered timestamps/measurements and Y is
            the number of classes, representing the cost of misclassifying a time series at each time step. For each
            matrix, a value at coordinates [i,j] represents the cost of predicting class j when the actual class is i.
            Usually, diagonals of the matrices are all zeros, and the costs between each matrix should globally
            increase. These costs should be defined by the domain expert.
            The function "create_cost_matrices" can be used to obtain these matrices from a single misclassification
            matrix, a function that outputs the cost of delay given the number of measurements in a time series and
            the set of timestamps.
        models_input_lengths: numpy.ndarray
            Input lengths considered by the trigger model. The time series' number of measurements for which a
            transition matrix and confusion matrix and thresholds are computed. These lengths should be the same as the
            ones used to train the chronological classifiers.
        nb_intervals: int, default=5
            Number of groups to aggregate the training time series into for each input length during the learning of the
            trigger model. The optimal value of this hyperparameter may depend on the task.
        aggregation_function: function/string. default='max'
            Function to aggregate the probabilities of each class for each time series prediction in the multiclass
            case to create the intervals thresholds. Known aggregation functions are listed in the
            'KNOWN_AGGREGATIONS' constant.
            See paper [2] for more detail.

    Attributes:
        thresholds: numpy.ndarray
            Array of shape (T, nb_intervals-1) where T is the number of input_lengths known by the trigger model.
            Thresholds correspond to the values between which the first class probability (or aggregated probabilities
            in the multiclass case) need to fall for a prediction to be aggregated to a certain group (interval).
            See paper [1] for more details.
        transition_matrices: numpy.ndarray
            Markov chain matrices of shape (T-1, nb_intervals, nb_intervals) where T is the number of input_lengths
            considered by the trigger model and where each row t contains the transition matrix of intervals from
            timestamp t to timestamp t+1.
            Each value [i,j] in the row t represents the probability for a time series to belonging to group j at t+1
            given that it belonged to group i at t.
            See paper [1] for more details.
        confusion_matrices: numpy.ndarray
            Array of shape (T, nb_intervals, Y, Y) where T is the number of input_lengths known by the trigger model and
            Y the number of classes. Contains for each input length the confusion matrix of the time series contained in
            each interval, with values [i, j] representing the probabilities of being classified in class j when the
            actual series' class is i.
        classes_: numpy.ndarray
            Array showing the names and order of classes, necessary to interpret the predictions contained in the
            training set.
        initial_cost: float
            Cost returned when trying to obtain the delay and misclassification cost associated to the prediction of a
            time series whose length is not known by the trigger model (not contained in 'models_input_lengths') and
            below that of the first known length.
            Returns 0 for the the delay_cost summed with the cost of predicting the class with the highest prior
            probability found during training.
        multiclass: boolean
            Boolean representing whether the problem is multiclass or not. Influences whether the aggregation functions
            will be used or not.
            See paper [2] for more detail.

    [1] Early classification of time series, cost-based optimization criterion and algorithms - Y. Achenchabe, A. Bondu,
    A. Cornuejols, A. Dachraoui - Machine Learning 2021
    [2] Early Classification of Time Series: Cost-based multiclass Algorithms - P.E. Zafar, Y. Achenchabe, A. Bondu,
    A. Cornuejols, V. Lemaire - DSAA 2021
    """

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 nb_intervals=None,
                 aggregation_function='max',
                 split_k=None,
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.nb_intervals = nb_intervals
        self.aggregation_function = aggregation_function
        self.split_k = split_k
        self.n_jobs = n_jobs

        self.thresholds = None
        self.transition_matrices = None
        self.confusion_matrices = None
        self.classes_ = None
        self.initial_cost = None
        self.multiclass = None

    def get_params(self):
        return {
            "cost_matrices": self.cost_matrices,
            "models_input_lengths": self.models_input_lengths,
            "nb_intervals": self.nb_intervals,
            "aggregation_function": self.aggregation_function,
            "thresholds": self.thresholds,
            "transition_matrices": self.transition_matrices,
            "confusion_matrices": self.confusion_matrices,
            "classes_": self.classes_,
            "initial_cost": self.initial_cost,
            "multiclass": self.multiclass,
        }
    
    def _get_confusion_matrices(self, X_probas, X_intervals, y, nb_intervals):

        X_class = X_probas.argmax(axis=-1).T
        confusion_matrices = [] # shape (T, K, P, P)
        for t, timestamp_data in enumerate(X_class):
            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) 
                          for k in range(nb_intervals)]
            confusion = np.array(
                [confusion_matrix(y_true, y_pred, normalize='all', labels=self.classes_)
                for y_pred, y_true in group_data]
            )
            confusion_matrices.append(confusion)

        # Fix nans created by division by 0 when not enough data
        confusion_matrices = np.nan_to_num(
            confusion_matrices, nan=1/np.square(len(self.classes_))
        )
        return confusion_matrices

    def _get_transitions_matrices(self, X_intervals, nb_intervals):

        # Obtain transition_matrices, Shape (T-1, K, K)
        transition_matrices = []
        for t in range(len(self.models_input_lengths)-1):
            transition_matrices.append(
                np.array(
                    [np.array([np.sum((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(nb_intervals)]) /
                        np.sum(X_intervals[t] == i) for i in range(nb_intervals)]
                )
            )
        # fix nans created by division by 0 when not enough data
        transition_matrices = np.nan_to_num(transition_matrices, nan=1/nb_intervals)

        return transition_matrices
    
    def _get_costs(self, groups, timestamp, nb_intervals):

        timestamp_idx = np.nonzero(self.models_input_lengths == timestamp)[0][0]
        t_star = -1

        for group in groups:
            costs = []
            # If series length is not covered return initial cost? Else estimate cost
            if timestamp < np.min(self.models_input_lengths):
                costs.append(self.initial_cost)
                timestamp = np.min(self.models_input_lengths)
                gamma = np.repeat(1/nb_intervals, nb_intervals)
            else:
                gamma = np.zeros(nb_intervals)  # interval membership probability vector
                gamma[group] = 1

            for t in range(timestamp_idx, len(self.models_input_lengths)):
                
                cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.cost_matrices[t])
                costs.append(cost)
                if t != len(self.models_input_lengths) - 1:
                    gamma = np.matmul(gamma, self.transition_matrices[t])  # Update interval markov probability

            timestamp_idx += 1
            if np.argmin(costs) == 0:
                t_star = timestamp_idx - 1 
                break
            
        return np.array(costs), t_star
    
    def fit(self, X, X_probas, y, classes_=None):
        """
        Fit the trigger model based on the predictions probabilities obtained by classifying time series of progressive
        lengths (using predict_proba on a second training set with a ChronologicalClassifier object).
        Parameters:
            X_pred: np.ndarray
                Array of shape (N, T, P) with N the number of time series, T the number of input lengths that were
                used for prediction and P the predicted class probabilities vectors.
            y: np.ndarray
                Array of N labels corresponding to the N time series predicted in X_pred.
            classes_:
                Array showing the names and order of classes, necessary to interpret the predictions contained in the
                X_pred if they are not in order and the labels in y if they are non numerical.
        """

        # DATA VALIDATION / INTEGRITY
        # models_input_lengths
        if isinstance(self.models_input_lengths, list):
            self.models_input_lengths = np.array(self.models_input_lengths)
        elif not isinstance(self.models_input_lengths, np.ndarray):
            raise TypeError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
        if self.models_input_lengths.ndim != 1:
            raise ValueError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
        for l in self.models_input_lengths:
            if l < 0:
                raise ValueError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
        if len(np.unique(self.models_input_lengths)) != len(self.models_input_lengths):
            self.models_input_lengths = np.unique(self.models_input_lengths)
            warn("Removed duplicates timestamps in argument 'models_input_lengths'.")
        if self.models_input_lengths.shape[0] != len(self.cost_matrices):
            raise ValueError("Argument 'cost_matrices' should have as many matrices as there are values in argument "
                             "'models_input_lengths'.")
        
        if not callable(self.aggregation_function):
            if not isinstance(self.aggregation_function, str):
                raise TypeError("Argument 'aggregation_function' should either be a string or a callable function.")
            if self.aggregation_function.lower() not in KNOWN_AGGREGATIONS.keys():
                raise ValueError(f"Function {self.aggregation_function} from argument 'aggregation_function' "
                                 f"is not known. Known aggregations are {KNOWN_AGGREGATIONS.keys()}.")

        # X_probas
        X_probas = copy.deepcopy(X_probas)
        y = copy.deepcopy(y)
        if isinstance(X_probas, list):
            X_probas = np.array(X_probas)
        elif isinstance(X_probas, pd.DataFrame):
            X_probas = X_probas.to_numpy()
        elif not isinstance(X_probas, np.ndarray):
            raise TypeError("X_pred should be a 3-dimensional list, array or DataFrame of shape (N, T, P) with N the "
                            "number of time series, T the number of input lengths that were used for prediction and P "
                            "the predicted class probabilities vectors.")
        if X_probas.ndim != 3:
            raise ValueError("X_pred should be a 3-dimensional list, array or DataFrame of shape (N, T, P) with N the "
                            "number of time series, T the number of input lengths that were used for prediction and P "
                            "the predicted class probabilities vectors.")

        if X_probas.shape[2] != self.cost_matrices.values.shape[2]:
            raise ValueError("X_probas probability vectors should have the same number of classes as the "
                             "cost matrices.")
        if len(X_probas) == 0:
            raise ValueError("Dataset 'X_probas' to fit trigger_model on is empty.")
        for i in range(len(X_probas)):
            if len(X_probas[i]) != len(X_probas[0]):
                raise ValueError("The number of timestamps should be the same for all examples.")

        # y
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be an array of classes of size N with N the number of examples in X_probas")
        if len(y) != len(X_probas):
            raise ValueError("y should be an array of classes of size N with N the number of examples in X_probas.")
        if y.ndim != 1:
            raise ValueError("y should be an array of classes of size N with N the number of examples in X_probas.")
        
        if isinstance(classes_, list):
            classes_ = np.array(classes_)
        """
        if not isinstance(classes_, np.ndarray):
            raise TypeError("Argument classes_ should be a list of class labels in the order "
                            "of the probabilities in X_pred.")
        """

        # ASSIGNMENTS
        self.models_input_lengths = np.sort(self.models_input_lengths)

        if not callable(self.aggregation_function):
            self.aggregation_function = KNOWN_AGGREGATIONS[self.aggregation_function]
        else:
            self.aggregation_function = self.aggregation_function

        # Initialize classes_, initial_class and multiclass
        self.classes_ = np.array(classes_) if classes_ is not None else np.arange(np.min(y), np.max(y)+1)
        X_pred = X_probas.argmax(axis=-1)
        y = np.array([np.nonzero(self.classes_ == y_)[0][0] for y_ in y])
        prior_class_prediction = np.argmax(np.unique(y, return_counts=True)[1])

        # TODO: make sure there is no pathological behaviour linked to using initial_cost
        self.initial_cost = np.sum(
            np.unique(y, return_counts=True)[1]/y.shape[0] *
            self.cost_matrices[0,:,prior_class_prediction]
        )
        self.multiclass = False if len(self.classes_) <= 2 else True

        # Aggregate values if multiclass : shape (N, T), values = aggregated value or 1st class proba
        X_aggregated = np.apply_along_axis(
            self.aggregation_function, 2, X_probas
        ) if self.multiclass else X_probas[:, :, 0]

        # Obtain thresholds for each group : X_sorted shape (T, N), values = sorted aggregated value
        # Grid-search over values of K if param not given
        X_sorted = np.sort(X_aggregated.T)
        if isinstance(self.nb_intervals, int):
            k_candidates = [self.nb_intervals]
        elif isinstance(self.nb_intervals, list):
            k_candidates = self.nb_intervals
        else:
            max_candidates = np.minimum(11, (np.sqrt(len(X))).astype(int)+1)
            #max_candidates = 21
            k_candidates = np.arange(1, max_candidates)
        
        if len(k_candidates) == 0:
            k_candidates = [1]
        
        if self.split_k:
            if len(k_candidates) > 1:
                idx_sorted, idx_meta = train_test_split(
                    list(range(len(X_sorted.T))), train_size=self.split_k, stratify=y, random_state=42
                )      
                X_aggregated, X_aggregated_meta = X_aggregated[idx_sorted, :], X_aggregated[idx_meta, :]
                X_sorted, X_meta = X_sorted[:, idx_sorted], X_sorted[:, idx_meta]
                X_probas, X_proba_meta = X_probas[idx_sorted, :, :], X_probas[idx_meta, :, :]
                y, y_meta = y[idx_sorted], y[idx_meta]
            else:
                warn("split_k attribute is not None, but only one candidates given for k," 
                     " using all training data to learn the model")
        
        opt_costs = np.inf
        for k in  k_candidates:
            thresholds_indices = np.linspace(0, X_sorted.shape[1], k+1)[1:-1].astype(int)
            self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)

            # Get intervals given threshold : shape (T, N), values = group ID of each pred
            X_intervals = np.array(
                [[np.sum([threshold <= x for threshold in self.thresholds[i]]) 
                for x in timestamp_data] for i, timestamp_data in enumerate(X_aggregated.T)],
            dtype=int)

            self.transition_matrices = self._get_transitions_matrices(X_intervals, k)
            self.confusion_matrices = self._get_confusion_matrices(X_probas, X_intervals, y, k)
            
            #mean_costs = np.mean(
            #    [self._get_costs(group, self.models_input_lengths[0], k)[0]
            #     for group in X_intervals.T])
            
            all_t_star_idx = [self._get_costs(group, self.models_input_lengths[0], k)[1] for group in X_intervals.T]
            costs_tmp = [self.cost_matrices[t][X_pred[i, t]][y[i]] for i, t in enumerate(all_t_star_idx)]
            mean_costs = np.mean(costs_tmp)
            
            if mean_costs < opt_costs:
                opt_costs = mean_costs
                self.nb_intervals = k
        
        if len(k_candidates) > 1:
            if self.split_k:
                X_sorted, X_probas, X_aggregated, y = X_meta, X_proba_meta, X_aggregated_meta, y_meta

            thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_intervals+1)[1:-1].astype(int)
            self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)
            X_intervals = np.array(
                [[np.sum([threshold <= x for threshold in self.thresholds[i]]) 
                for x in timestamp_data] for i, timestamp_data in enumerate(X_aggregated.T)]
            )
            self.transition_matrices = self._get_transitions_matrices(X_intervals, self.nb_intervals)
            self.confusion_matrices = self._get_confusion_matrices(X_probas, X_intervals, y, self.nb_intervals)

        return self

    def predict(self, X, X_probas):

        """
        Given a list of N predicted class probabilities vectors, predicts whether it is the right time to trigger the
        prediction as well as the expected costs of delaying the decisions. Predictions whose time series' input length
        is not known by the trigger model are approximated to the last known length. In the case where it is below the
        first known length, the 'initial_cost' attribute, based on prior probabilities, is returned.

        Parameters:
            X_pred: numpy.ndarray
                Array of N predicted class probabilities vectors obtained by predicting time series of different lengths.
            predicted_series_lengths: numpy.ndarray
                Array of int representing the input lengths of the time series associated to each predictions. Allows
                the trigger model to determine which thresholds, transition matrices and confusions matrices to use.

        Returns:
            triggers: np.ndarray
                Array of booleans representing whether it is the right time to trigger the decision for each predicted
                time series.
            costs: np.ndarray
                Array of arrays representing the forecasted cost of delaying the decision until the number of
                measurements reaches each one of the input lengths known by the trigger model for each time series
                prediction.
        """
        # DATA VALIDATION / INTEGRITY
        # X_probas
        X_probas = check_X_probas(X_probas)

        # PREPARE DATA FOR PREDICTION
        # Update predicted_series_lengths to compatible lengths
        truncated = False
        timestamps = []
        for ts in X:
            diff = self.models_input_lengths - len(ts)
            if 0 not in diff:
                if (diff > 0).sum() == len(diff):
                    timestamps.append(0)
                    continue
                else:
                    time_idx = np.where(diff > 0, diff, -np.inf).argmax()
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
            timestamps.append(self.models_input_lengths[time_idx])

        # Aggregate probas : shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(
            self.aggregation_function, 1, X_probas
        ) if self.multiclass else X_probas[:, 0]

        # Get intervals given threshold : shape (N), values = group
        X_intervals = []
        for i, x in enumerate(X_aggregated):
            if timestamps[i] < np.min(self.models_input_lengths):
                interval = np.nan
            else:
                interval = np.sum(
                    [threshold <= x for threshold in self.thresholds[
                    np.nonzero(self.models_input_lengths == timestamps[i])[0][0]]], dtype=int
                )
                X_intervals.append(interval)

        # CALCULATE AND RETURN COSTS
        triggers, costs = [], []
        returned_priors = False
        for n, group in enumerate(X_intervals):
            prediction_forecasted_costs = self._get_costs([group], timestamps[n], self.nb_intervals)[0]
            prediction_forecasted_trigger = True if np.argmin(prediction_forecasted_costs) == 0 else False
            triggers.append(prediction_forecasted_trigger)
            costs.append(prediction_forecasted_costs)

        # Send warnings and return
        if truncated:
            warn("Some predictions lengths were unknown to the trigger model. Predictions were approximated using the "
                 "last known length.")
        if returned_priors:
            warn("Some predictions lengths where below that of the first length known by the trigger model. "
                 "Cost of predicting the most frequent class in priors was used.")
            
        return np.array(triggers), costs


class StoppingRule(TriggerModel):

    """
    A myopic trigger model which triggers a decision whenever a stopping rule representing time and confidence passes a
    certain threshold. Trigger models aim to determine the best time to trigger a decision given evolving time series.
    Introduced in :
    Early classification of time series by simultaneously optimizing the accuracy and earliness - U. Mori, A. Mendiburu,
    S. Dasgupta, J. A. Lozano - IEEE transactions on neural networks and learning systems
    """

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 stopping_rule="SR1",
                 objective="avg_cost",
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.stopping_rule = stopping_rule
        self.objective = objective
        self.n_jobs = n_jobs

    def _trigger(self, gammas, probas, t):

        if self.stopping_rule == "SR1":
            proba1 = np.max(probas) 
            proba2 = np.min(np.delete(np.abs(probas - proba1), probas.argmax())) 
            probas = np.array([proba1, proba2])

        score = gammas[:-1] @ probas + gammas[-1] * (t/self.max_length)

        if score > 0 or t==self.models_input_lengths[-1]:
            return True

        return False 

    def _get_score(self, gammas, X_probas, y):

        costs = []
        for i in range(len(X_probas)):
            for j, t in enumerate(self.models_input_lengths):
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
            earliness = 1 - np.mean([c[1] for c in costs]) / np.max(self.cost_matrices.delay_cost)
            agg_cost = -hmean((acc, earliness)) # minimize inverse 
        else:
            raise ValueError("Unknown objective function, should be one of ['avg_cost', 'hmean']")

        return agg_cost

    def fit(self, X, X_probas, y, classes_=None):
        
        self.max_length = X.shape[1]

        nb_gammas = 3 if self.stopping_rule == "SR1" else X_probas.shape[2]+1
        self.candidates_gammas = list(permutations(np.linspace(-1, 1, 11), nb_gammas))

        gamma_costs = Parallel(n_jobs=self.n_jobs) \
                    (delayed(self._get_score)(gammas, X_probas, y) for gammas in self.candidates_gammas)
        
        self.opt_gammas = self.candidates_gammas[np.argmin(gamma_costs)]

        return self

    def predict(self, X, X_probas):
        
        timestamps = []
        for ts in X:
            diff = self.models_input_lengths - len(ts)
            if 0 not in diff:
                truncated = True
                time_idx = np.where(diff > 0, diff, -np.inf).argmax()
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
            timestamps.append(self.models_input_lengths[time_idx])
        
        triggers = []
        for i, p in enumerate(X_probas):
            triggers.append(
                self._trigger(self.opt_gammas, p, timestamps[i])
            )

        return np.array(triggers), None


class TEASER(TriggerModel):

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 objective='hmean',
                 n_jobs=1):
        
        super().__init__()

        ######Constant attributes#######
        self.require_past_probas = True
        ################################

        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.objective = objective

        self.n_jobs = n_jobs

    def _generate_features(self, probas):

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        preds = probas.argmax(axis=-1)

        features = np.concatenate(
            (probas, preds[:, None], diff[:, None]), axis=-1
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
                cv=min(len(oc_features[masks_pos_probas]), 5),
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
    
    def fit(self, X, X_probas, y, classes_=None):
        
        self.max_length = X.shape[1]
        self.n_classes = len(np.unique(y))

        # get predictions for each timestamp
        X_pred = X_probas.argmax(axis=-1)
        masks_pos_probas = np.array([x==y for x in X_pred.T])

        res = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._fit_master_clf)(X_probas[:,j,:], masks_pos_probas[j]) 
             for j in range(len(self.models_input_lengths)))
        self.master_clfs, X_oc = zip(*res)

        # search best threshold v
        best_obj = np.inf
        for v in np.arange(1, 6):
            final_pred = np.ones(len(X)) * (-1)
            final_t_star = copy.deepcopy(final_pred)
            consecutive_pred = np.zeros(len(X))

            for j, t in enumerate(self.models_input_lengths):
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
                if t == self.models_input_lengths[-1]:
                    final_pred = np.where(final_pred == -1, current_pred, final_pred)
                    final_t_star = np.where(final_t_star == -1, t, final_t_star)
            
            if self.objective == 'hmean':
                acc = (final_pred == y).mean()
                earliness = 1 - (np.mean(final_t_star) / self.max_length)
                # minimize inverse of highest hmean
                if best_obj > -hmean((acc, earliness)):
                    best_obj = -hmean((acc, earliness))
                    self.best_v = v
            else:
                t_idx = [
                    np.where(self.models_input_lengths == final_t_star[i])[0][0] 
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
    
    def predict(self, X, X_probas):

        timestamps = []
        for ts in X:
            diff = self.models_input_lengths - len(ts)
            if 0 not in diff:
                time_idx = np.where(diff > 0, diff, -np.inf).argmax()
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
            timestamps.append(self.models_input_lengths[time_idx])

        triggers = []
        for i, probas in enumerate(X_probas):
            if timestamps[i] < self.best_v:
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

        return np.array(triggers), None


class ECEC(TriggerModel):

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 objective="avg_cost",
                 n_jobs=1):
        
        super().__init__()

        ######Constant attributes#######
        self.require_past_probas = True
        ################################

        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
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
        for j in range(len(self.models_input_lengths)):

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

            confidences = self._get_fused_confidence(X_pred[i])
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
            earliness = 1 - np.mean([c[1] for c in costs]) / np.max(self.cost_matrices.delay_cost)
            agg_cost = -hmean((acc, earliness)) # minimize inverse 
        else:
            raise ValueError("Unknown objective function, should be one of ['avg_cost', 'hmean']")

        return agg_cost

    def fit(self, X, X_probas, y, classes_=None):

        X_pred = X_probas.argmax(axis=-1)

        # compute all ratios for all classifiers
        self.ratios = []
        for j in range(len(self.models_input_lengths)):
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
    
    def predict(self, X, X_probas):
        
        triggers = []
        for probas in X_probas:
            preds = np.array(probas).argmax(axis=-1)
            confidence = self._get_fused_confidence(preds)[-1]

            if confidence >= self.opt_threshold:
                triggers.append(True)
            else:
                triggers.append(False)

        return np.array(triggers), None


class ECDIRE(TriggerModel):

    def __init__(self,
                 chronological_classifiers,
                 threshold_acc=1.,
                 cross_validation=False,
                 n_jobs=1):

        super().__init__()

        ######Constant attributes#######
        self.alter_classifiers = True
        ################################

        self.models_input_lengths = copy.deepcopy(chronological_classifiers.models_input_lengths)
        self.chronological_classifiers = copy.deepcopy(chronological_classifiers)
        self.chronological_classifiers.prev_models_input_lengths = self.models_input_lengths

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
                 for j, t in enumerate(clfs.models_input_lengths)]
        probs = [clfs.classifiers[j].predict_proba(X[test_idx, :t]) 
                 for j, t in enumerate(clfs.models_input_lengths)]
        
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
            timestamp = self.models_input_lengths[timestamp_idx]
            safe_timestamps = [t[0] for t in timeline] if len(timeline) > 0 else []

            if timestamp in safe_timestamps:
                timeline[safe_timestamps.index(timestamp)][1].add(c)
            else:
                timeline.append((timestamp, {c}))

        return sorted(timeline)
    
    def _get_reliability(self, probas):

        thresholds = []
        for j in range(len(self.models_input_lengths)):
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
        
    def fit(self, X, X_probas, y, classes_=None):

        if self.cross_validation:
            rskf = RepeatedStratifiedKFold(random_state=4)
            results_cv = Parallel(n_jobs=self.n_jobs) \
                (delayed(self._fit_cv)(X, y, train_idx, test_idx) for train_idx, test_idx in rskf.split(X, y)) 
        
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
        if max_t != self.models_input_lengths[-1]:
            idx = np.where(self.models_input_lengths == max_t)[0][0] + 1
            self.timeline.extend(
                ((t, set(y)) for t in self.models_input_lengths[idx:])
            )
        else:
            self.timeline[-1] = (max_t, set(y))
        
        clf_idx = [np.where(self.models_input_lengths==t)[0][0]
                   for t in list(zip(*self.timeline))[0]]
        
        self.reliability = self.reliability[clf_idx]
        self.models_input_lengths = self.models_input_lengths[clf_idx]
        self.chronological_classifiers.models_input_lengths = self.models_input_lengths
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
    
    def predict(self, X, X_probas):

        timestamps = []
        for ts in X:
            diff = self.models_input_lengths - len(ts)
            if 0 not in diff:
                truncated = True
                if (diff > 0).sum() == len(diff):
                    timestamps.append(0)
                else:
                    time_idx = np.where(diff > 0, diff, -np.inf).argmax()
                    timestamps.append(self.models_input_lengths[time_idx])
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
                timestamps.append(self.models_input_lengths[time_idx])

        triggers = []
        for i, probas in enumerate(X_probas):
            trigger  = False
            if timestamps[i] == 0:
                triggers.append(trigger)
            else:
                pred = probas.argmax(axis=-1)
                safe_class = [c for t, c in self.timeline
                              if timestamps[i] == t]
                if pred in safe_class[0]:
                    probas_diff = np.array([np.max(probas)-x for x in probas])
                    probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
                    probas_diff = np.min(probas_diff)

                    t_idx = np.where(self.models_input_lengths == timestamps[i])
                    if probas_diff >= self.reliability[t_idx[0][0], pred]:
                        trigger = True
                triggers.append(trigger)

        return np.array(triggers), None


class CALIMERA(TriggerModel):
    """
    CALIMERA: A new early time series classification method
    Inspired by : https://github.com/JakubBilski/CALIMERA 
    """
    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths

        self.n_jobs = n_jobs
    
    def _generate_features(self, probas, time_idx, y):

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        #delay_cost = self.alpha * self.models_input_lengths[time_idx] / self.max_length
        #costs = 1 - max_probas + delay_cost
        
        proba_correct = np.mean(np.diagonal(self.cost_matrices[time_idx])) * max_probas
        non_diag = self.cost_matrices[time_idx] - \
            (np.eye(self.n_classes) * np.diagonal(self.cost_matrices[time_idx]))
        proba_incorrect = np.sum(non_diag) / (self.n_classes**2 - self.n_classes) * (1-max_probas)
        costs = proba_correct + proba_incorrect

        return features, costs
    
    def fit(self, X, X_probas, y, classes_=None):

        self.max_length = X.shape[1]
        self.max_timestamp = len(self.models_input_lengths)
        self.n_classes = len(np.unique(y))

        results  = [self._generate_features(X_probas[:,t,:], t, y) 
                    for t in range(X_probas.shape[1])]
        
        features, costs = zip(*results)
        features, costs = (np.array(features), np.array(costs))

        self.halters = [None for _ in range(self.max_timestamp-1)]

        for t in range(self.max_timestamp-2, -1, -1):

            X_trigger = features[t]
            y_trigger = costs[t+1] - costs[t] 

            model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
            self.halters[t] = model
            predicted_cost_difference = model.predict(X_trigger)
            for j in range(len(X_trigger)):
                if predicted_cost_difference[j] < 0:
                    costs[t, j] = costs[t+1, j]

        return self 
    
    def predict(self, X, X_probas):

        timestamps = []
        for ts in X:
            diff = self.models_input_lengths - len(ts)
            if 0 not in diff:
                if (diff > 0).sum() == len(diff):
                    timestamps.append(0)
                    continue
                else:
                    time_idx = np.where(diff > 0, diff, -np.inf).argmax()
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
            timestamps.append(self.models_input_lengths[time_idx])

        triggers, costs = [], []
        for i, probas in enumerate(X_probas):
            trigger = False
            # if last timestamp is reached
            if timestamps[i] == self.models_input_lengths[-1]:
                triggers.append(True)
                costs.append(np.nan)
                continue

            time_idx = np.where(timestamps[i] == self.models_input_lengths)[0][0]
            X_trigger, _ = self._generate_features(probas[None,:], time_idx, 
                                                   np.zeros(X.shape[0], dtype=int))
            
            predicted_cost_diff = self.halters[time_idx].predict(X_trigger)

            if predicted_cost_diff > 0:
                trigger = True
            
            triggers.append(trigger)
            costs.append(predicted_cost_diff[0])


        return np.array(triggers), np.array(costs)