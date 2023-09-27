import copy
import numpy as np
import pandas as pd

from warnings import warn
from itertools import permutations
from collections.abc import Iterable
from joblib import Parallel, delayed
from scipy.stats import hmean
from sklearn.svm import OneClassSVM
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from utils import check_X_probas

def gini(probas):
    return 1 - np.sum(np.square(probas))

KNOWN_AGGREGATIONS = {"max": np.max, "gini": gini}

class ProbabilityThreshold:

    def __init__(self,
                cost_matrices,
                models_input_lengths,
                manual_threshold=None,
                n_jobs=1):
        
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths

        if manual_threshold is not None:
            self.opt_threshold = manual_threshold
        
        self.n_jobs = n_jobs
    
    def _get_score(self, threshold, X_probas, y):
        
        costs = []
        for i, probas in enumerate(X_probas):
            for j in range(len(self.models_input_lengths)):
                trigger = (np.max(probas[j]) >= threshold)
                if trigger:
                    pred = np.argmax(probas[j])
                    costs.append(
                        self.cost_matrices[j][pred][y[i]]
                    )
                    break

                if j==len(self.models_input_lengths)-1:
                    pred = np.argmax(probas[j])
                    costs.append(
                        self.cost_matrices[j][pred][y[i]]
                    )

        if isinstance(costs[0], np.void):
            minus_acc  = np.mean([c[0] for c in costs])
            earliness = np.mean([c[1] for c in costs])
            agg_cost = hmean((minus_acc, earliness))
        else:
            agg_cost = np.mean(costs)

        return agg_cost
        
    def fit(self, X, X_probas, y, classes_=None):
        
        if hasattr(self, 'opt_threshold'):
            return self
        
        n_classes = len(np.unique(y))
        candidate_threshold = np.linspace(1/n_classes, 1, 21)

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

        
class EconomyGamma:
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
                 nb_intervals=5,
                 aggregation_function='max'):

        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.nb_intervals = nb_intervals
        self.aggregation_function = aggregation_function
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
        # cost_matrices
        if isinstance(self.cost_matrices, list):
            self.cost_matrices = np.array(self.cost_matrices)
        elif not isinstance(self.cost_matrices, np.ndarray):
            raise TypeError(
                "Argument 'cost_matrices' should be an array of shape (T,Y,Y) with T the number of timestamps and Y"
                "the number of classes.")
        if self.cost_matrices.ndim != 3 or self.cost_matrices.shape[1] != self.cost_matrices.shape[2]:
            raise ValueError(
                "Argument 'cost_matrices' should be an array of shape (T,Y,Y) with T the number of timestamps and Y"
                "the number of classes.")

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
        if self.models_input_lengths.shape[0] != self.cost_matrices.shape[0]:
            raise ValueError("Argument 'cost_matrices' should have as many matrices as there are values in argument "
                             "'models_input_lengths'.")

        # nb_intervals
        if not isinstance(self.nb_intervals, int):
            raise TypeError("Argument nb_intervals should be a strictly positive int.")
        if self.nb_intervals < 1:
            raise ValueError("Argument nb_intervals should be a strictly positive int.")
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

        if X_probas.shape[2] != self.cost_matrices.shape[1]:
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
        X_sorted = np.sort(X_aggregated.T)
        thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_intervals + 1)[1:-1].astype(int)
        self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)

        # Get intervals given threshold : shape (T, N), values = group ID of each pred
        X_intervals = np.array(
            [[np.sum([threshold <= x for threshold in self.thresholds[i]]) 
              for x in timestamp_data] for i, timestamp_data in enumerate(X_aggregated.T)]
        )

        # Obtain transition_matrices, Shape (T-1, K, K)
        self.transition_matrices = []
        for t in range(len(self.models_input_lengths) - 1):
            self.transition_matrices.append(
                np.array(
                    [np.array([np.sum((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(self.nb_intervals)]) /
                        np.sum(X_intervals[t] == i) for i in range(self.nb_intervals)]
                )
            )
        # fix nans created by division by 0 when not enough data
        self.transition_matrices = np.nan_to_num(self.transition_matrices, nan=1/self.nb_intervals)

        # Obtain confusion matrices : X_class shape (T, N) values = class of each prediction
        X_class = np.apply_along_axis(np.argmax, 2, X_probas).T
        self.confusion_matrices = [] # shape (T, K, P, P)
        for t, timestamp_data in enumerate(X_class):

            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) 
                          for k in range(self.nb_intervals)]
            confusion = np.array(
                [np.array(
                    [[np.sum((x_ == np.nonzero(self.classes_ == j)[0][0]) & (y_ == np.nonzero(self.classes_ == i)[0][0]))
                      for j in self.classes_] for i in self.classes_]
                ) / len(y_) for x_, y_ in group_data]
            )
            self.confusion_matrices.append(confusion)

        # Fix nans created by division by 0 when not enough data
        self.confusion_matrices = np.nan_to_num(
            self.confusion_matrices, nan=1/np.square(len(self.classes_))
        )

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
                truncated = True
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
                interval = np.sum([threshold <= x 
                                   for threshold in self.thresholds[
                                       np.nonzero(self.models_input_lengths == timestamps[i])[0][0]
                                    ]]
                            )
                X_intervals.append(interval)

        # CALCULATE AND RETURN COSTS
        triggers, costs = [], []
        returned_priors = False
        for n, group in enumerate(X_intervals):

            prediction_forecasted_costs = []

            # If series length is not covered return initial cost? Else estimate cost
            if timestamps[n] < np.min(self.models_input_lengths):
                prediction_forecasted_costs.append(self.initial_cost)
                timestamps[n] = np.min(self.models_input_lengths)
                gamma = np.repeat(1 / self.nb_intervals, self.nb_intervals)
                returned_priors = True
            else:
                gamma = np.zeros(self.nb_intervals)  # interval membership probability vector
                gamma[group] = 1

            # Estimate cost for each length from prediction length to max_length
            for t in range(np.nonzero(self.models_input_lengths == timestamps[n])[0][0],
                           len(self.models_input_lengths)):
                
                cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.cost_matrices[t])
                prediction_forecasted_costs.append(cost)
                if t != len(self.models_input_lengths) - 1:
                    gamma = np.matmul(gamma, self.transition_matrices[t])  # Update interval markov probability

            # Save estimated costs and determine trigger time
            prediction_forecasted_costs = np.array(prediction_forecasted_costs)
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


class StoppingRule:

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
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.stopping_rule = stopping_rule
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
                    costs.append(self.cost_matrices[j][pred][y[i]])
                    break
        
        if isinstance(costs[0], np.void):
            minus_acc  = np.mean([c[0] for c in costs])
            earliness = np.mean([c[1] for c in costs])
            agg_cost = hmean((minus_acc, earliness))
        else:
            agg_cost = np.mean(costs)

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


class TEASER:

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 objective='hmean',
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.objective = objective

        self.n_jobs = n_jobs

    def fit(self, X, X_probas, y, classes_=None):
        
        self.max_length = X.shape[1]

        # get predictions for each timestamp
        X_pred = X_probas.argmax(axis=-1)
        masks_pos_probas = np.array([x==y for x in X_pred.T])

        self.master_clfs, X_oc = [], []
        for j in range(len(self.models_input_lengths)):
            probas = X_probas[:,j,:]
            pred = probas.argmax(axis=-1)

            probas_diff = np.array([np.max(x)-x for x in probas])
            probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
            probas_diff = np.min(probas_diff, axis=-1)

            oc_features = np.concatenate(
                (probas, pred[:, None], probas_diff[:, None]),
                axis=-1
            )
            X_oc.append(oc_features)

            oc_clf = OneClassSVM(kernel='rbf', nu=.05, tol=1e-4)
            gamma_grid = (
                {"gamma": [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]}
            )
            gs = GridSearchCV(
                estimator=oc_clf,
                param_grid=gamma_grid,
                scoring='accuracy',
                cv=min(len(oc_features), 5),
                n_jobs=self.n_jobs
            )
            # train the set of master classifiers only
            # on positives samples 
            gs.fit(oc_features[masks_pos_probas[j]], 
                   np.ones(len(oc_features[masks_pos_probas[j]]))
            )
            self.master_clfs.append(gs.best_estimator_)

        # search best threshold v
        best_obj = -1
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
                
                # if final timestep is reached samples that hasn't been 
                # triggered yet are trigger 
                if t == self.models_input_lengths[-1]:
                    final_pred = np.where(final_pred == -1, current_pred, final_pred)
                    final_t_star = np.where(final_t_star == -1, t, final_t_star)
            
            if self.objective == 'hmean':
                acc = (final_pred == y).mean()
                earliness = 1 - (np.mean(final_t_star) / self.max_length)

                if best_obj < hmean((acc, earliness)):
                    best_obj = hmean((acc, earliness))
                    self.best_v = v
            else:
                t_idx = [
                    np.where(self.models_input_lengths == final_t_star[i])[0][0] 
                    for i in range(len(final_t_star))
                ]
                avg_cost = np.mean(
                    [self.cost_matrices[int(t_idx[i])-1][int(p)][y[i]]
                     for i, p in enumerate(final_pred)]
                )
                if best_obj < avg_cost:
                    best_obj = avg_cost
                    self.best_v = v

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
        for i, probas in enumerate(X_probas):
            if timestamps[i] < self.best_v:
                triggers.append(False)
            else:
                if isinstance(probas, list):
                    pred = np.array(probas).argmax(axis=-1)
                    probas_diff = np.array([np.max(x)-x for x in probas])
                    probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
                    probas_diff = np.min(probas_diff, axis=-1)
                else:
                    pred = np.array([probas.argmax(axis=-1)])
                    probas_diff = np.array([np.max(probas)-x for x in probas])
                    probas_diff = np.where(probas_diff==0, np.inf, probas_diff)
                    probas_diff = np.array([np.min(probas_diff, axis=-1)])
                    probas = probas[None, :]

                oc_features = np.concatenate(
                    (probas, pred[:, None], probas_diff[:, None]),
                    axis=-1
                )
                accept = [(self.master_clfs[l].predict(oc_features[l:l+1]) == 1)
                          for l in range(len(probas))]
                accept = np.array(accept)
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


class ECEC:

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 n_jobs=1):
        
        super().__init__()
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.n_jobs = n_jobs
        
    def _get_ratio(self, preds, y):

        ratios = []
        classes = np.unique(y)

        for y1 in classes:
            for y2 in classes:
                nominator = len(preds[(preds == y2) & (y == y1)])
                denominator = len(preds[(preds == y2)])

                ratios.append(nominator / denominator)              

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
                    costs.append(self.cost_matrices[j][pred][y[i]])
                    break
        
        if isinstance(costs[0], np.void):
            minus_acc  = np.mean([c[0] for c in costs])
            earliness = np.mean([c[1] for c in costs])
            agg_cost = hmean((minus_acc, earliness))
        else:
            agg_cost = np.mean(costs)

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


class ECDIRE:

    def __init__(self,
                 chronological_classifiers,
                 threshold_acc=1.,
                 cross_validation=False,
                 n_jobs=1):

        super().__init__()

        self.models_input_lengths = chronological_classifiers.models_input_lengths
        self.chronological_classifiers = chronological_classifiers

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
            timestamp_idx = np.where((acc_class < acc_threshold))[0][-1] + 1
            timeline.append(
                (self.models_input_lengths[timestamp_idx], {c})
            )

        return sorted(timeline)
    
    def _get_reliability(self, probas):

        thresholds = []
        for j in range(len(self.models_input_lengths)):
            probas_t = np.vstack([clf[j] for clf in probas])
            
            class_thresholds = []
            for c in range(probas_t.shape[-1] - 1):
                probas_c = probas_t[(probas_t[:,-1] == c)][:, :-1]

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


class CALIMERA:

    def __init__(self,
                 cost_matrices,
                 models_input_lengths,
                 alpha,
                 n_jobs=1):
        
        self.cost_matrices = cost_matrices
        self.models_input_lengths = models_input_lengths
        self.alpha = alpha

        self.n_jobs = n_jobs
    
    def _generate_features(self, probas, time_idx, y):

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]

        features = np.concatenate(
            (probas, max_probas[:,None], second_max_probas[:,None]), axis=-1
        )
        delay_cost = self.alpha * self.models_input_lengths[time_idx] / self.max_length
        costs = 1 - max_probas + delay_cost

        #preds = probas.argmax(axis=-1)
        #costs = [self.cost_matrices[time_idx][pred][y[i]]
        #         for i, pred in enumerate(preds)]

        return features, costs
    
    def fit(self, X, X_probas, y, classes_=None):

        self.max_length = X.shape[1]
        self.max_timestamp = len(self.models_input_lengths)

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
                else:
                    time_idx = np.where(diff > 0, diff, -np.inf).argmax()
                    timestamps.append(self.models_input_lengths[time_idx])
            else:
                time_idx = np.where(diff == 0, diff, -np.inf).argmax()
                timestamps.append(self.models_input_lengths[time_idx])

        triggers, costs = [], []
        trigger = False
        for i, probas in enumerate(X_probas):
            time_idx = np.where(timestamps[i] == self.models_input_lengths)[0][0]
            X_trigger, _ = self._generate_features(probas[None,:], time_idx, 
                                                   np.zeros(X.shape[0], dtype=int))
            
            predicted_cost_diff = self.halters[time_idx].predict(X_trigger)

            if predicted_cost_diff > 0:
                trigger = True
            
            triggers.append(trigger)
            costs.append(predicted_cost_diff[0])


        return np.array(triggers), np.array(costs)