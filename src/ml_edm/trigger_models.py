import numpy as np
import pandas as pd
from warnings import warn
import copy


def gini(probas):
    return 1 - np.sum(np.square(probas))


KNOWN_AGGREGATIONS = {"max": np.max, "gini": gini}


def create_cost_matrices(timestamps, misclassification_cost, delay_cost=None):
    """
    A function that converts a separated misclassification matrix and a delay cost function to an array of cost_matrices
    showing the evolution of the misclassification cost along time given a series of timestamps for EconomyGamma.

    Parameters:
        timestamps: numpy.ndarray
            Array of int representing the timestamps / time series input length, to create cost_matrices for.
        misclassification_cost: numpy.ndarray
            Array of size Y*Y where Y is the number of classes and where each value at indices
            [i,j] represents the cost of predicting class j when the actual class is i. Usually, diagonals of the
            matrix are all zeros. This cost must be defined by a domain expert and be expressed in the same unit
            as the delay cost.
        delay_cost: python function, default=None
            Function that takes as input a time series input length and returns the timely cost of waiting to obtain
            such number of measurements given the task. This cost must be defined by a domain expert and be expressed
            in the same unit as the misclassification cost.
    """

    # INPUT VALIDATION
    # models_input_lengths
    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)
    elif not isinstance(timestamps, np.ndarray):
        raise TypeError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
    if timestamps.ndim != 1:
        raise ValueError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
    for l in timestamps:
        if l < 0:
            raise ValueError("Argument 'models_input_lengths' should be a 1D-array of positive int.")
    if len(np.unique(timestamps)) != len(timestamps):
        timestamps = np.unique(timestamps)
        warn("Removed duplicates timestamps in argument 'models_input_lengths'.")

    # misclassification_cost
    if isinstance(misclassification_cost, list):
        misclassification_cost = np.array(misclassification_cost)
    elif not isinstance(misclassification_cost, np.ndarray):
        raise TypeError(
            "Argument 'misclassification_cost' should be an array of shape (Y,Y) with Y the number of clases.")
    if misclassification_cost.ndim != 2 or misclassification_cost.shape[0] != \
            misclassification_cost.shape[1]:
        raise ValueError(
            "Argument 'misclassification_cost' should be an array of shape (Y,Y) with Y the number of clases.")

    # delay_cost
    if delay_cost is None:
        return np.repeat(misclassification_cost[None, :], len(timestamps), axis=0)
    if not callable(delay_cost):
        raise TypeError("Argument delay_cost should be a function that returns a cost given a time series length.")

    # CREATE AND RETURN COST MATRICES
    return np.array([misclassification_cost + delay_cost(timestamp) for timestamp in timestamps])


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

    def fit(self, X_pred, y, classes_=None):
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
                "Argument 'cost_matrices' should be an array of shape (T, Y,Y) with T the number of timestamps and Y "
                "the number of classes.")
        if self.cost_matrices.ndim != 3 or self.cost_matrices.shape[1] != self.cost_matrices.shape[2]:
            raise ValueError(
                "Argument 'cost_matrices' should be an array of shape (T, Y,Y) with T the number of timestamps and Y "
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

        # X_pred
        X_pred = copy.deepcopy(X_pred)
        y = copy.deepcopy(y)
        if isinstance(X_pred, list):
            X_pred = np.array(X_pred)
        elif isinstance(X_pred, pd.DataFrame):
            X_pred = X_pred.to_numpy()
        elif not isinstance(X_pred, np.ndarray):
            raise TypeError("X_pred should be a 3-dimensional list, array or DataFrame of shape (N, T, P) with N the "
                            "number of time series, T the number of input lengths that were used for prediction and P "
                            "the predicted class probabilities vectors.")
        if X_pred.ndim != 3:
            raise ValueError("X_pred should be a 3-dimensional list, array or DataFrame of shape (N, T, P) with N the "
                            "number of time series, T the number of input lengths that were used for prediction and P "
                            "the predicted class probabilities vectors.")

        if X_pred.shape[2] != self.cost_matrices.shape[1]:
            raise ValueError("X_pred probability vectors should have the same number of classes as the "
                             "cost matrices.")
        if len(X_pred) == 0:
            raise ValueError("Dataset 'X_pred' to fit trigger_model on is empty.")
        for i in range(len(X_pred)):
            if len(X_pred[i]) != len(X_pred[0]):
                raise ValueError("The number of timestamps should be the same for all examples.")

        # y
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be an array of classes of size N with N the number of examples in X_pred")
        if len(y) != len(X_pred):
            raise ValueError("y should be an array of classes of size N with N the number of examples in X_pred.")
        if y.ndim != 1:
            raise ValueError("y should be an array of classes of size N with N the number of examples in X_pred.")
        if isinstance(classes_, list):
            classes_ = np.array(classes_)
        if not isinstance(classes_, np.ndarray):
            raise TypeError("Argument classes_ should be a list of class labels in the order "
                            "of the probabilities in X_pred.")

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
        self.initial_cost = \
            np.sum(np.unique(y, return_counts=True)[1]/y.shape[0] * self.cost_matrices[0,:,prior_class_prediction])
        self.multiclass = False if len(self.classes_) <= 2 else True

        # Aggregate values if multiclass : shape (N, T), values = aggregated value or 1st class proba
        X_aggregated = np.apply_along_axis(self.aggregation_function, 2, X_pred) if self.multiclass else X_pred[:, :, 0]

        # Obtain thresholds for each group : X_sorted shape (T, N), values = sorted aggregated value
        X_sorted = np.sort(X_aggregated.transpose())
        thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_intervals + 1)[1:-1].astype(int)
        self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)

        # Get intervals given threshold : shape (T, N), values = group ID of each pred
        X_intervals = \
            np.array([[np.count_nonzero([threshold <= x for threshold in self.thresholds[i]]) for j, x
                       in enumerate(timestamp_data)] for i, timestamp_data in enumerate(X_aggregated.transpose())])

        # Obtain transition_matrices, Shape (T-1, K, K)
        self.transition_matrices = []
        for t in range(len(self.models_input_lengths) - 1):
            self.transition_matrices.append(np.array([np.array([np.count_nonzero((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(self.nb_intervals)]) /
                                                      np.count_nonzero(X_intervals[t] == i) for i in range(self.nb_intervals)]))
        # fix nans created by division by 0 when not enough data
        self.transition_matrices = np.nan_to_num(self.transition_matrices, nan=1/self.nb_intervals)

        # Obtain confusion matrices : X_class shape (T, N) values = class of each prediction
        X_class = np.apply_along_axis(np.argmax, 2, X_pred).transpose()
        self.confusion_matrices = [] # shape (T, K, P, P)
        for t, timestamp_data in enumerate(X_class):
            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) for k in range(self.nb_intervals)]
            confusion = np.array([np.array([[np.count_nonzero((x_ == np.nonzero(self.classes_ == j)[0][0]) & (y_ == np.nonzero(self.classes_ == i)[0][0])) for j in self.classes_] for i in self.classes_])
                                  / len(y_) for x_, y_ in group_data])
            self.confusion_matrices.append(confusion)
        # Fix nans created by division by 0 when not enough data
        self.confusion_matrices = np.nan_to_num(self.confusion_matrices, nan=1 / np.square(len(self.classes_)))
        return self

    def predict(self, X_pred, predicted_series_lengths):
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
        # X_pred
        X_pred = copy.deepcopy(X_pred)
        if isinstance(X_pred, list):
            X_pred = np.array(X_pred)
        elif isinstance(X_pred, pd.DataFrame):
            X_pred = X_pred.to_numpy()
        elif not isinstance(X_pred, np.ndarray):
            raise TypeError(
                "X_pred should be a 2-dimensional list, array or DataFrame of size (N, P) with N the number "
                "of examples and P the number of classes probabilities.")
        if X_pred.ndim != 2:
            raise ValueError(
                "X_pred should be a 2-dimensional list, array or DataFrame of size (N, P) with N the number "
                "of examples and P the number of classes probabilities.")
        if len(X_pred) == 0:
            raise ValueError("Dataset 'X_pred' to predict triggering on is empty.")

        # predicted_series_lengths
        if isinstance(predicted_series_lengths, list):
            predicted_series_lengths = np.array(predicted_series_lengths)
        elif not isinstance(predicted_series_lengths, np.ndarray):
            raise TypeError("Argument 'predicted_series_lengths' should be an 1D-array of time series lengths from "
                            "which the predictions in X_pred where obtained.")
        if predicted_series_lengths.ndim != 1:
            raise ValueError("Argument 'predicted_series_lengths' should be an 1D-array of time series lengths from "
                            "which the predictions in X_pred where obtained.")

        # PREPARE DATA FOR PREDICTION
        # Update predicted_series_lengths to compatible lengths
        predicted_series_lengths = np.array(predicted_series_lengths)
        truncated = False
        for i, pred_length in enumerate(predicted_series_lengths):
            if pred_length not in self.models_input_lengths:
                for length in self.models_input_lengths[::-1]:
                    if length < pred_length:
                        predicted_series_lengths[i] = length
                        truncated = True
                        break

        # Aggregate probas : shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(self.aggregation_function, 1, X_pred) if self.multiclass else X_pred[:, 0]

        # Get intervals given threshold : shape (N), values = group
        X_intervals = []
        for i, x in enumerate(X_aggregated):
            if predicted_series_lengths[i] < np.min(self.models_input_lengths):
                interval = np.nan
            else:
                interval = np.count_nonzero([threshold <= x for threshold
                                             in self.thresholds[
                                                 np.nonzero(
                                                     self.models_input_lengths == predicted_series_lengths[i])[
                                                     0][0]]])
                X_intervals.append(interval)

        # CALCULATE AND RETURN COSTS
        triggers, costs = [], []
        returned_priors = False
        for n, group in enumerate(X_intervals):

            prediction_forecasted_costs = []

            # If series length is not covered return initial cost? Else estimate cost
            if predicted_series_lengths[n] < np.min(self.models_input_lengths):
                prediction_forecasted_costs.append(self.initial_cost)
                predicted_series_lengths[n] = np.min(self.models_input_lengths)
                gamma = np.repeat(1 / self.nb_intervals, self.nb_intervals)
                returned_priors = True

            else:
                gamma = np.zeros(self.nb_intervals)  # interval membership probability vector
                gamma[group] = 1

            # Estimate cost for each length from prediction length to max_length
            for t in range(np.nonzero(self.models_input_lengths == predicted_series_lengths[n])[0][0],
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
        return np.array(triggers), np.array(costs)


# TODO add stopping rule model
"""
class StoppingRule(TriggerModel):
    """"""
    A myopic trigger model which triggers a decision whenever a stopping rule representing time and confidence passes a
    certain threshold. Trigger models aim to determine the best time to trigger a decision given evolving time series.
    Introduced in :
    Early classification of time series by simultaneously optimizing the accuracy and earliness - U. Mori, A. Mendiburu,
    S. Dasgupta, J. A. Lozano - IEEE transactions on neural networks and learning systems
    """"""
    def __init__(self, earliness_cost_coeff=0):
        super().__init__()
        self.nb_groups = nb_groups
        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost
        self.earliness_cost_coeff = earliness_cost_coeff

    def fit(self, x_pred, y):
        return self

    def predict(self, x, timestamps):
        return trigger, cost"""