import numpy as np
from warnings import warn
from base_trigger_models import TriggerModel, NonMyopicTriggerModel

class EconomyGamma(NonMyopicTriggerModel):
    """
    A highly performing non-myopic trigger model based on the economy architecture for early time series classification.
    Allows the anticipation of future decision costs based on a supervised grouping of time series and user-defined
    costs for delay and misclassification. Trigger models aim to determine the best time to trigger a decision given
    evolving time series.
    Introduced in :
    Early classification of time series, cost-based optimization criterion and algorithms - Y. Achenchabe, A. Bondu,
    A. Cornuejols, A. Dachraoui - Machine Learning 2021
    """
    def __init__(self, misclassification_cost=None, delay_cost=None, nb_groups=0, trigger_model_series_lengths=None, aggregation_type='gini'):
        super().__init__()
        self.nb_groups = nb_groups
        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost
        self.aggregation_function = np.max
        self.multiclass = None
        self.classes_ = None
        self.trigger_model_series_lengths = np.array(trigger_model_series_lengths)
        self.transition_matrices = None  # shape (T-1, K, K)
        self.confusion_matrices = None  # shape (T, K, P, P)
        self.thresholds = None  # shape (T, K-1)


    def fit(self, X_pred, y, classes_=None):
        # shape (N, T, P), values = probability of each class according to the classifier
        self.classes_ = np.array(classes_) if classes_ is not None else np.arange(np.min(y), np.max(y)+1)
        y = np.array([np.nonzero(self.classes_ == y_)[0][0] for y_ in y])
        prior_class_prediction = np.argmax(np.unique(y, return_counts=True)[1])
        self.initial_cost = np.sum(np.unique(y, return_counts=True)[1]/y.shape[0] * self.misclassification_cost[:,prior_class_prediction]) + self.delay_cost(0)
        self.multiclass = False if len(self.classes_) <= 2 else True
        # shape (N, T), values = aggregated value or 1st class proba
        X_aggregated = np.apply_along_axis(self.aggregation_function, 2, X_pred) if self.multiclass else X_pred[:, :, 0]
        # shape (T, N), values = sorted aggregated value
        X_sorted = np.sort(X_aggregated.transpose())
        thresholds_indices = np.linspace(0, X_sorted.shape[1], self.nb_groups+1)[1:-1].astype(int)
        self.thresholds = X_sorted[:, thresholds_indices]
        # shape (T, N), values = group ID of each pred
        # argsort seems to be broken ?
        #X_intervals = np.floor_divide(np.argsort(X_aggregated.transpose(), axis=1), X_pred.shape[0] / self.nb_groups)
        X_intervals = np.array([[np.count_nonzero([threshold <= x for threshold in self.thresholds[i]]) for j, x in enumerate(timestamp_data)] for i, timestamp_data in enumerate(X_aggregated.transpose())])
        self.transition_matrices = []
        for t in range(len(self.trigger_model_series_lengths) - 1):
            self.transition_matrices.append(np.array([np.array([np.count_nonzero((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(self.nb_groups)]) /
                                                      np.count_nonzero(X_intervals[t] == i) for i in range(self.nb_groups)]))
        self.transition_matrices = np.nan_to_num(self.transition_matrices, nan=1/self.nb_groups)  # fix nans created by division by 0 when not enough data
        # shape (T, N) values = class of each prediction
        X_class = np.apply_along_axis(np.argmax, 2, X_pred).transpose()
        self.confusion_matrices = []
        for t, timestamp_data in enumerate(X_class):
            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) for k in range(self.nb_groups)]
            confusion = np.array([np.array([[np.count_nonzero((x_ == np.nonzero(self.classes_ == j)[0][0]) & (y_ == np.nonzero(self.classes_ == i)[0][0])) for j in self.classes_] for i in self.classes_]) / len(y_) for x_, y_ in group_data])
            self.confusion_matrices.append(confusion)
        self.confusion_matrices = np.nan_to_num(self.confusion_matrices, nan=1 / np.square(len(self.classes_)))  # fix nans created by division by 0 when not enough data
        return self

    def predict(self, X_pred, predicted_series_lengths):
        #for now supposes the timestamps in corresponding_timestamps correspond to the timestamps learned in fit. TODO: deal with this <
        predicted_series_lengths = np.array(predicted_series_lengths)
        truncated = False
        for i, pred_length in enumerate(predicted_series_lengths):
            if pred_length not in self.trigger_model_series_lengths:
                for length in self.trigger_model_series_lengths[::-1]:
                    if length < pred_length:
                        predicted_series_lengths[i] = length
                        truncated = True
                        break
        # TODO deal with pred_length < trigger_model_series_lengths[0] and write warnings / delay cost takes lengths as input??
        # shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(self.aggregation_function, 1, X_pred) if self.multiclass else X_pred[:, 0]
        # shape (N), values = group
        X_intervals = []
        for i, x in enumerate(X_aggregated):
            if predicted_series_lengths[i] < np.min(self.trigger_model_series_lengths):
                interval = np.nan
            else:
                interval = np.count_nonzero([threshold <= x for threshold
                                  in self.thresholds[
                                      np.nonzero(self.trigger_model_series_lengths == predicted_series_lengths[i])[0][
                                          0]]])
                X_intervals.append(interval)
        triggers, costs, forecasted_triggers, forecasted_costs = [], [], [], []
        returned_priors = False
        for n, group in enumerate(X_intervals):
            prediction_forecasted_costs = []
            if predicted_series_lengths[n] < np.min(self.trigger_model_series_lengths):
                prediction_forecasted_costs.append(self.initial_cost)
                predicted_series_lengths[n] = np.min(self.trigger_model_series_lengths)
                gamma = np.repeat(1/self.nb_groups, self.nb_groups)
                returned_priors = True
            else:
                gamma = np.zeros(self.nb_groups)
                gamma[group] = 1
            # for each length from prediction length to max_length
            for t in range(np.nonzero(self.trigger_model_series_lengths == predicted_series_lengths[n])[0][0], len(self.trigger_model_series_lengths)):
                cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.misclassification_cost) + self.delay_cost(self.trigger_model_series_lengths[t])
                prediction_forecasted_costs.append(cost)
                if t != len(self.trigger_model_series_lengths) - 1:
                    gamma = np.matmul(gamma, self.transition_matrices[t])
            prediction_forecasted_costs = np.array(prediction_forecasted_costs)
            prediction_forecasted_trigger = np.argmin(prediction_forecasted_costs)
            prediction_cost = prediction_forecasted_costs[0]
            prediction_trigger = True if prediction_forecasted_trigger == 0 else False
            triggers.append(prediction_trigger)
            costs.append(prediction_cost)
            forecasted_triggers.append(prediction_forecasted_trigger)
            forecasted_costs.append(prediction_forecasted_costs)
            if truncated:
                warn("Some predictions lengths were unknown to the trigger model. Last known length was assumed")
            if returned_priors:
                warn("Some predictions lengths where below that of the first length known by the trigger model. "
                              "Cost of predicting the most frequent class in priors was used.")
        return np.array(triggers), np.array(costs), np.array(forecasted_triggers), np.array(forecasted_costs)


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