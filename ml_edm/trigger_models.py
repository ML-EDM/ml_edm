import numpy as np
import pandas as pd
from warnings import warn
import copy


def gini(probas):
    return 1 - np.sum(np.square(probas))


KNOWN_AGGREGATIONS = {"max": np.max, "gini": gini}


class EconomyGamma():
    """
    A highly performing non-myopic trigger model based on the economy architecture for early time series classification.
    Allows the anticipation of future decision costs based on a supervised grouping of time series and user-defined
    costs for delay and misclassification. Trigger models aim to determine the best time to trigger a decision given
    evolving time series.
    Introduced in :
    Early classification of time series, cost-based optimization criterion and algorithms - Y. Achenchabe, A. Bondu,
    A. Cornuejols, A. Dachraoui - Machine Learning 2021
    """

    def __init__(self,
                 misclassification_cost,
                 delay_cost,
                 models_series_lengths,
                 nb_intervals=5,
                 aggregation_function='max'):

        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost
        self.models_series_lengths = models_series_lengths
        self.nb_intervals = nb_intervals
        self.aggregation_function = aggregation_function

    def get_params(self):
        return {
            "misclassification_cost": self.misclassification_cost,
            "delay_cost": self.delay_cost,
            "models_series_lengths": self.models_series_lengths,
            "nb_intervals": self.nb_intervals,
            "aggregation_function": self.aggregation_function,
            "thresholds": self.thresholds,
            "transition_matrices": self.transition_matrices,
            "confusion_matrices": self.confusion_matrices,
            "classes_": self.classes_,
            "multiclass": self.multiclass,
            "initial_cost": self.initial_cost,
        }

    # Xpred shape (N, T, P), values = probability of each class according to the classifier
    def fit(self, X_pred, y, classes_=None):

        # DATA VALIDATION
        if isinstance(self.misclassification_cost, list):
            self.misclassification_cost = np.array(self.misclassification_cost)
        elif not isinstance(self.misclassification_cost, np.ndarray):
            raise TypeError(
                "Argument 'misclassification_cost' should be an array of shape (Y,Y) with Y the number of clases.")
        if self.misclassification_cost.ndim != 2 or self.misclassification_cost.shape[0] != self.misclassification_cost.shape[1]:
            raise ValueError(
                "Argument 'misclassification_cost' should be an array of shape (Y,Y) with Y the number of clases.")
        if self.misclassification_cost is None:
            raise ValueError("Argument delay_cost is missing.")
        if not callable(self.delay_cost):
            raise TypeError("Argument delay_cost should be a function that returns a cost given a time series length.")
        if isinstance(self.models_series_lengths, list):
            self.models_series_lengths = np.array(self.models_series_lengths)
        elif not isinstance(self.models_series_lengths, np.ndarray):
            raise TypeError("Argument 'models_series_lengths' should be a 1D-array of positive int.")
        if self.models_series_lengths.ndim != 1:
            raise ValueError("Argument 'models_series_lengths' should be a 1D-array of positive int.")
        for l in self.models_series_lengths:
            if l < 0:
                raise ValueError("Argument 'models_series_lengths' should be a 1D-array of positive int.")
        if len(np.unique(self.models_series_lengths)) != len(self.models_series_lengths):
            self.models_series_lengths = np.unique(self.models_series_lengths)
            warn("Removed duplicates timestamps in argument 'models_series_lengths'.")
        if not isinstance(self.nb_intervals, int):
            raise TypeError("Argument nb_intervals should be a strictly positive int.")
        if self.nb_intervals < 1:
            raise ValueError("Argument nb_intervals should be a strictly positive int.")
        if not callable(self.aggregation_function):
            if not isinstance(self.aggregation_function, str):
                raise TypeError("Argument 'aggregation_function' should either be a string or a callable function.")
            if self.aggregation_function.lower() not in KNOWN_AGGREGATIONS.keys():
                raise ValueError(f"Function {self.aggregation_function} from argument 'aggregation_function' is not known. "
                                 f"Known aggregations are {KNOWN_AGGREGATIONS.keys()}.")
        # Check X_pred and y
        X_pred = copy.deepcopy(X_pred)
        y = copy.deepcopy(y)
        if isinstance(X_pred, list):
            X_pred = np.array(X_pred)
        elif isinstance(X_pred, pd.DataFrame):
            X_pred = X_pred.to_numpy()
        elif not isinstance(X_pred, np.ndarray):
            raise TypeError("X_pred should be a 3-dimensional list, array or DataFrame of size (N, T, Z) with N the number "
                            "of examples, T the number of timestamps and Z the number of classes probabilities.")
        if X_pred.ndim != 3:
            raise ValueError("X_pred should be a 3-dimensional list, array or DataFrame of size (N, T, Z) with N the number "
                             "of examples, T the number of timestamps and Z the number of classes probabilities.")
        if len(X_pred) == 0:
            raise ValueError("Dataset 'X_pred' to fit trigger_model on is empty.")
        for i in range(len(X_pred)):
            if len(X_pred[i]) != len(X_pred[0]):
                raise ValueError("The number of timestamps should be the same for all examples.")
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
            raise TypeError("Argument classes_ should be a list of class labels in the oreder of the probabilities in X_pred")

        # ASSIGNMENTS
        self.models_series_lengths = np.sort(self.models_series_lengths)
        if not callable(self.aggregation_function):
            self.aggregation_function = KNOWN_AGGREGATIONS[self.aggregation_function]
        else:
            self.aggregation_function = self.aggregation_function

        # Initialize classes_, initial_class and multiclass
        self.classes_ = np.array(classes_) if classes_ is not None else np.arange(np.min(y), np.max(y)+1)
        y = np.array([np.nonzero(self.classes_ == y_)[0][0] for y_ in y])
        prior_class_prediction = np.argmax(np.unique(y, return_counts=True)[1])
        # TODO: verify there is no pathological behaviour linked to using initial_cost
        self.initial_cost = np.sum(np.unique(y, return_counts=True)[1]/y.shape[0] * self.misclassification_cost[:,prior_class_prediction]) + self.delay_cost(0)
        self.multiclass = False if len(self.classes_) <= 2 else True

        # Aggregate values if multiclass : shape (N, T), values = aggregated value or 1st class proba
        X_aggregated = np.apply_along_axis(self.aggregation_function, 2, X_pred) if self.multiclass else X_pred[:, :, 0]

        # Obtain thresholds for each group : X_sorted shape (T, N), values = sorted aggregated value
        X_sorted = np.sort(X_aggregated.transpose())
        thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_intervals + 1)[1:-1].astype(int)
        self.thresholds = X_sorted[:, thresholds_indices] # shape (T, K-1)

        # Get intervals given threshold : shape (T, N), values = group ID of each pred
        X_intervals = np.array([[np.count_nonzero([threshold <= x for threshold in self.thresholds[i]]) for j, x in enumerate(timestamp_data)] for i, timestamp_data in enumerate(X_aggregated.transpose())])

        # Obtain transition_matrices
        self.transition_matrices = [] # shape (T-1, K, K)
        for t in range(len(self.models_series_lengths) - 1):
            self.transition_matrices.append(np.array([np.array([np.count_nonzero((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(self.nb_intervals)]) /
                                                      np.count_nonzero(X_intervals[t] == i) for i in range(self.nb_intervals)]))
        self.transition_matrices = np.nan_to_num(self.transition_matrices, nan=1/self.nb_intervals)  # fix nans created by division by 0 when not enough data

        # Obtain confusion matrices : X_class shape (T, N) values = class of each prediction
        X_class = np.apply_along_axis(np.argmax, 2, X_pred).transpose()
        self.confusion_matrices = [] # shape (T, K, P, P)
        for t, timestamp_data in enumerate(X_class):
            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) for k in range(self.nb_intervals)]
            confusion = np.array([np.array([[np.count_nonzero((x_ == np.nonzero(self.classes_ == j)[0][0]) & (y_ == np.nonzero(self.classes_ == i)[0][0])) for j in self.classes_] for i in self.classes_]) / len(y_) for x_, y_ in group_data])
            self.confusion_matrices.append(confusion)
        self.confusion_matrices = np.nan_to_num(self.confusion_matrices, nan=1 / np.square(len(self.classes_)))  # fix nans created by division by 0 when not enough data
        return self

    def predict(self, X_pred, predicted_series_lengths):
        X_pred = copy.deepcopy(X_pred)
        if isinstance(X_pred, list):
            X_pred = np.array(X_pred)
        elif isinstance(X_pred, pd.DataFrame):
            X_pred = X_pred.to_numpy()
        elif not isinstance(X_pred, np.ndarray):
            raise TypeError(
                "X_pred should be a 2-dimensional list, array or DataFrame of size (N, Z) with N the number "
                "of examples and Z the number of classes probabilities.")
        if X_pred.ndim != 2:
            raise ValueError(
                "X_pred should be a 2-dimensional list, array or DataFrame of size (N, T, Z) with N the number "
                "of examples, T the number of timestamps and Z the number of classes probabilities.")
        if len(X_pred) == 0:
            raise ValueError("Dataset 'X_pred' to fit trigger_model on is empty.")
        if isinstance(predicted_series_lengths, list):
            predicted_series_lengths = np.array(predicted_series_lengths)
        elif not isinstance(predicted_series_lengths, np.ndarray):
            raise TypeError("Argument 'predicted_series_lengths' should be an 1D-array of time series lengths from which "
                            "the predictions in X_pred where obtained.")
        if predicted_series_lengths.ndim != 1:
            raise TypeError("Argument 'predicted_series_lengths' should be an 1D-array of time series lengths from which "
                            "the predictions in X_pred where obtained.")

        # Update predicted_series_lengths to compatible lengths
        predicted_series_lengths = np.array(predicted_series_lengths)
        truncated = False
        for i, pred_length in enumerate(predicted_series_lengths):
            if pred_length not in self.models_series_lengths:
                for length in self.models_series_lengths[::-1]:
                    if length < pred_length:
                        predicted_series_lengths[i] = length
                        truncated = True
                        break

        # Aggregate probas : shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(self.aggregation_function, 1, X_pred) if self.multiclass else X_pred[:, 0]

        # Get intervals given threshold : shape (N), values = group
        X_intervals = []
        for i, x in enumerate(X_aggregated):
            if predicted_series_lengths[i] < np.min(self.models_series_lengths):
                interval = np.nan
            else:
                interval = np.count_nonzero([threshold <= x for threshold
                                             in self.thresholds[
                                                 np.nonzero(
                                                     self.models_series_lengths == predicted_series_lengths[i])[
                                                     0][0]]])
                X_intervals.append(interval)

        # Calculate and return costs
        triggers, costs = [], []
        returned_priors = False
        for n, group in enumerate(X_intervals):

            prediction_forecasted_costs = []

            # If series length is not covered return initial cost
            if predicted_series_lengths[n] < np.min(self.models_series_lengths):
                prediction_forecasted_costs.append(self.initial_cost)
                predicted_series_lengths[n] = np.min(self.models_series_lengths)
                gamma = np.repeat(1 / self.nb_intervals, self.nb_intervals)
                returned_priors = True

            else:
                gamma = np.zeros(self.nb_intervals)
                gamma[group] = 1

            # Estimate cost for each length from prediction length to max_length
            for t in range(np.nonzero(self.models_series_lengths == predicted_series_lengths[n])[0][0], len(self.models_series_lengths)):
                misclassification_cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.misclassification_cost)
                delay_cost = self.delay_cost(self.models_series_lengths[t])
                prediction_forecasted_costs.append(misclassification_cost + delay_cost)
                if t != len(self.models_series_lengths) - 1:
                    gamma = np.matmul(gamma, self.transition_matrices[t])  # Update interval markov probability

            # Save estimated costs and determine trigger time
            prediction_forecasted_costs = np.array(prediction_forecasted_costs)
            prediction_forecasted_trigger = True if np.argmin(prediction_forecasted_costs) == 0 else False
            triggers.append(prediction_forecasted_trigger)
            costs.append(prediction_forecasted_costs)

        # Send warnings and return
        if truncated:
            warn("Some predictions lengths were unknown to the trigger model. Last known length was assumed")
        if returned_priors:
            warn("Some predictions lengths where below that of the first length known by the trigger model. "
                 "Cost of predicting the most frequent class in priors was used.")
        return np.array(triggers), np.array(costs)



"""
    def predict_on_first_trigger(self, x_pred, timestamps):
        return trigger_time, cost, forecasted_costs

    def predict_on_optimal_cost(self, x_pred, y, timestamps):
        return trigger_time, trigger, cost, forecasted_trigger, forecasted_costs"""


"""class EconomyK(NonMyopicTriggerModel):
    """"""
    A non-myopic trigger model based on the economy architecture for early time series classification. Allows the
    anticipation of future decision costs based on an unsupervised grouping of time series with kmeans and user-defined
    costs for delay and misclassification. Trigger models aim to determine the best time to trigger a decision given
    evolving time series.
    Introduced in :
    Early classification of time series, cost-based optimization criterion and algorithms - Y. Achenchabe, A. Bondu,
    A. Cornuejols, A. Dachraoui - Machine Learning 2021
    """"""
    def __init__(self, nb_groups=0, misclassification_cost, delay_cost):
        super().__init__()
        self.nb_groups = nb_groups
        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost

    def fit(self, x, x_preds, y):
        return self

    def predict(self, x, timestamps):
        return trigger, cost, forecasted_trigger, forecasted_costs

    def predict_on_first_trigger(self, x, timestamps):
        return trigger_time, cost, forecasted_costs

    def predict_on_optimal_cost(self, x, y, timestamps):
        return trigger_time, trigger, cost, forecasted_trigger, forecasted_costs


class StoppingRule(TriggerModel):
    """"""
    # TODO
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
        return trigger, cost

    def predict_on_first_trigger(self, x, timestamps):
        return trigger_time, cost

    def predict_on_optimal_cost(self, x, y, timestamps):
        return trigger_time, trigger, cost"""