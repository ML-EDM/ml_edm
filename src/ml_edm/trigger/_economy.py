import numpy as np
from warnings import warn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ._base import BaseTriggerModel

def gini(probas):
    return 1 - np.sum(np.square(probas))

KNOWN_AGGREGATIONS = {"max": np.max, "gini": gini}

class EconomyGamma(BaseTriggerModel):
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
                 timestamps,
                 nb_intervals=None,
                 aggregation_function='max',
                 split_k=None,
                 n_jobs=1):
        
        super().__init__(timestamps)

        self.nb_intervals = nb_intervals
        self.aggregation_function = aggregation_function
        self.split_k = split_k
        self.n_jobs = n_jobs

        self.thresholds = None
        self.transition_matrices = None
        self.confusion_matrices = None
        self.initial_cost = None
        self.multiclass = None

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
        for t in range(len(self.timestamps)-1):
            transition_matrices.append(
                np.array(
                    [np.array([np.sum(
                        (X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(nb_intervals)]) /
                        np.sum(X_intervals[t] == i) for i in range(nb_intervals)]
                )
            )
        # fix nans created by division by 0 when not enough data
        transition_matrices = np.nan_to_num(transition_matrices, nan=1/nb_intervals)

        return transition_matrices
    
    def _get_costs(self, groups, timestamp, nb_intervals):

        returned_priors = False
        timestamp_idx = np.nonzero(self.timestamps == timestamp)[0][0]
        t_star = -1

        for group in groups:
            costs = []
            # If series length is not covered return initial cost? Else estimate cost
            if timestamp < np.min(self.timestamps):
                costs.append(self.initial_cost)
                timestamp = np.min(self.timestamps)
                gamma = np.repeat(1/nb_intervals, nb_intervals)
                returned_priors = True
            else:
                gamma = np.zeros(nb_intervals)  # interval membership probability vector
                gamma[group] = 1

            for t in range(timestamp_idx, len(self.timestamps)):
                
                cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.cost_matrices[t].T)
                costs.append(cost)
                if t != len(self.timestamps) - 1:
                    gamma = np.matmul(gamma, self.transition_matrices[t])  # Update interval markov probability

            timestamp_idx += 1
            if np.argmin(costs) == 0:
                t_star = timestamp_idx - 1 
                break
        
        if returned_priors:
            warn("Some predictions lengths where below that of the first length known by the trigger model. "
                 "Cost of predicting the most frequent class in priors was used.")
            
        return np.array(costs), t_star
    
    def _fit(self, X_probas, y):
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

        # aggregation function for multiclass 
        if not callable(self.aggregation_function):
            if not isinstance(self.aggregation_function, str):
                raise TypeError("Argument 'aggregation_function' should either be a string or a callable function.")
            if self.aggregation_function.lower() not in KNOWN_AGGREGATIONS.keys():
                raise ValueError(f"Function {self.aggregation_function} from argument 'aggregation_function' "
                                 f"is not known. Known aggregations are {KNOWN_AGGREGATIONS.keys()}.")

        if not callable(self.aggregation_function):
            self.aggregation_function = KNOWN_AGGREGATIONS[self.aggregation_function]
        else:
            self.aggregation_function = self.aggregation_function

        # Initialize classes_, initial_class and multiclass
        X_pred = X_probas.argmax(axis=-1)
        #y = np.array([np.nonzero(self.classes_ == y_)[0][0] for y_ in y]) LabelEncoder()

        y_counts = np.unique(y, return_counts=True)[1]
        prior_class_prediction = np.argmax(y_counts)

        # TODO: make sure there is no pathological behaviour linked to using initial_cost
        self.initial_cost = np.sum(
            y_counts / y.shape[0] * self.cost_matrices[0, prior_class_prediction, :]
        )
        self.multiclass = False if len(self.classes_) <= 2 else True

        # Aggregate values if multiclass : shape (N, T), values = aggregated value or 1st class proba
        X_aggregated = np.apply_along_axis(
            self.aggregation_function, 2, X_probas
        ) if self.multiclass else X_probas[:, :, 0]

        # Obtain thresholds for each group : X_sorted shape (T, N), values = sorted aggregated value
        # Grid-search over values of K if param not given
        if isinstance(self.nb_intervals, int):
            k_candidates = [self.nb_intervals]
        elif isinstance(self.nb_intervals, list):
            k_candidates = self.nb_intervals
        else:
            max_candidates = np.minimum(11, (np.sqrt(len(X_probas))).astype(int)+1)
            #max_candidates = 21
            k_candidates = np.arange(1, max_candidates)
        
        if len(k_candidates) == 0:
            k_candidates = [1]

        if self.split_k:
            if len(k_candidates) > 1:
                idx_val, idx_meta = train_test_split(
                    list(range(len(X_aggregated))), train_size=self.split_k, stratify=y, random_state=42
                )      
                X_aggregated, X_aggregated_meta = X_aggregated[idx_val, :], X_aggregated[idx_meta, :]
                #X_sorted, X_meta = X_sorted[:, idx_val], X_sorted[:, idx_meta]
                X_probas, X_proba_meta = X_probas[idx_val, :, :], X_probas[idx_meta, :, :]
                y, y_meta = y[idx_val], y[idx_meta]
            else:
                warn("split_k attribute is not None, but only one candidates given for k," 
                     " using all training data to learn the model")
        
        X_sorted = np.sort(X_aggregated.T)
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
            
            all_t_star_idx = [self._get_costs(group, self.timestamps[0], k)[1] for group in X_intervals.T]
            costs_tmp = [self.cost_matrices[t][X_pred[i, t]][y[i]] for i, t in enumerate(all_t_star_idx)]
            mean_costs = np.mean(costs_tmp)
            
            if mean_costs < opt_costs:
                opt_costs = mean_costs
                self.nb_intervals = k
        
        if len(k_candidates) > 1:
            if self.split_k:
                X_probas, X_aggregated, y = X_proba_meta, X_aggregated_meta, y_meta
                X_sorted = np.sort(X_aggregated.T)

            thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_intervals+1)[1:-1].astype(int)
            self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)
            X_intervals = np.array(
                [[np.sum([threshold <= x for threshold in self.thresholds[i]]) 
                for x in timestamp_data] for i, timestamp_data in enumerate(X_aggregated.T)]
            )
            self.transition_matrices = self._get_transitions_matrices(X_intervals, self.nb_intervals)
            self.confusion_matrices = self._get_confusion_matrices(X_probas, X_intervals, y, self.nb_intervals)

        return self

    def _predict(self, X_probas, X_timestamps):
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
        # Aggregate probas : shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(
            self.aggregation_function, 1, X_probas
        ) if self.multiclass else X_probas[:, 0]

        # Get intervals given threshold : shape (N), values = group
        X_intervals = []
        for i, x in enumerate(X_aggregated):
            if X_timestamps[i] < np.min(self.timestamps):
                interval = np.nan
            else:
                interval = np.sum(
                    [threshold <= x for threshold in self.thresholds[
                    np.nonzero(self.timestamps == X_timestamps[i])[0][0]]], dtype=int
                )
                X_intervals.append(interval)

        # CALCULATE AND RETURN COSTS
        triggers, self.forecast_costs = [], []
        for n, group in enumerate(X_intervals):
            prediction_forecasted_costs = self._get_costs([group], X_timestamps[n], self.nb_intervals)[0]
            prediction_forecasted_trigger = True if np.argmin(prediction_forecasted_costs) == 0 else False
            triggers.append(prediction_forecasted_trigger)
            self.forecast_costs.append(prediction_forecasted_costs)
            
        return np.array(triggers)