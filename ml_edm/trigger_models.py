import numpy as np

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
    def __init__(self, misclassification_cost=None, delay_cost=None, nb_groups=0, aggregation_type='gini'):
        super().__init__()
        self.nb_groups = nb_groups
        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost
        self.aggregation_function = np.max
        self.multiclass = None
        self.time_series_length = None
        self.transition_matrices = None  # shape (T-1, K, K)
        self.confusion_matrices = None  # shape (T, K, P, P)
        self.thresholds = None  # shape (T, K-1)


    def fit(self, X_pred, y):
        # shape (N, T, P), values = probability of each class according to the classifier
        self.time_series_max_length = X_pred.shape[1]
        self.nb_class = len(np.unique(y))
        self.multiclass = False if self.nb_class <= 2 else True
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
        for t in range(self.time_series_max_length - 1):
            self.transition_matrices.append(
                np.array([np.array([np.count_nonzero((X_intervals[t+1] == j) & (X_intervals[t] == i)) for j in range(self.nb_groups)]) /
                          np.count_nonzero(X_intervals[t] == i) for i in range(self.nb_groups)]))
        self.transition_matrices = np.array(self.transition_matrices)
        # shape (T, N) values = class of each prediction
        X_class = np.apply_along_axis(np.argmax, 2, X_pred).transpose()
        self.confusion_matrices = []
        for t, timestamp_data in enumerate(X_class):
            group_data = [(timestamp_data[X_intervals[t] == k], y[X_intervals[t] == k]) for k in range(self.nb_groups)]
            confusion = np.array([np.array([[np.count_nonzero((x_ == j) & (y_ == i)) for j in range(self.nb_class)] for i in range(self.nb_class)]) / len(y_) for x_, y_ in group_data])
            self.confusion_matrices.append(confusion)
        self.confusion_matrices = np.array(self.confusion_matrices)
        return self

    def predict(self, X_pred, corresponding_timestamps):
        #for now supposes the timestamps in time_series_length correspond to the timestamps learned in fit.
        # shape (N), values = aggregated pred
        X_aggregated = np.apply_along_axis(self.aggregation_function, 1, X_pred) if self.multiclass else X_pred[:, 0]
        # shape (N), values = group
        X_intervals = np.array([np.count_nonzero([threshold <= x for threshold in self.thresholds[corresponding_timestamps[i]]]) for i, x in enumerate(X_aggregated)])
        triggers, costs, forecasted_triggers, forecasted_costs = [], [], [], []
        for n, group in enumerate(X_intervals):
            prediction_forecasted_costs = []
            gamma = np.zeros(self.nb_groups)
            gamma[group] = 1
            for t in range(corresponding_timestamps[n], self.time_series_max_length):
                cost = np.sum(gamma[:,None,None] * self.confusion_matrices[t] * self.misclassification_cost) + self.delay_cost(t)
                prediction_forecasted_costs.append(cost)
                if t == self.time_series_max_length - 1:
                    break
                gamma = np.matmul(gamma, self.transition_matrices[t])
            prediction_forecasted_costs = np.array(prediction_forecasted_costs)
            prediction_forecasted_trigger = np.argmin(prediction_forecasted_costs) + corresponding_timestamps[n]
            prediction_cost = prediction_forecasted_costs[0]
            prediction_trigger = True if prediction_forecasted_trigger == corresponding_timestamps[n] else False
            triggers.append(prediction_trigger)
            costs.append(prediction_cost)
            forecasted_triggers.append(prediction_forecasted_trigger)
            forecasted_costs.append(prediction_forecasted_costs)
        return triggers, costs, forecasted_triggers, forecasted_costs


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