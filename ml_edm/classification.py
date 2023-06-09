import copy
import numpy as np
import pandas as pd

from dataset import extract_features, get_time_series_lengths
from trigger_models import EconomyGamma

from sklearn.ensemble import HistGradientBoostingClassifier

from warnings import warn


class ChronologicalClassifiers:
    """
    A class used to manage a list of classifiers trained on time series of incrementing lengths following Scikit-Learn
    conventions.
    """

    # TODO : Multivariate data
    # TODO : Change feature extraction
    # TODO : Add calibration
    # TODO : deal with NaN in dataset ? (Although HGBoost is nan compatible)
    # TODO : Add set_params, Add setters in every class, compatibility setter in Early Classifier
    # TODO : Optimize decision threshold
    # TODO : Verbose, loadbar?
    # TODO : implement sparse matrix compatibility
    # TODO : check_estimator scikitlearn -> make estimator
    # TODO : create factory code?

    def __init__(self,
                 nb_classifiers=None,
                 base_classifier=None,
                 learned_timestamps_ratio=None,  # learning_timestamps_ratio | learning_ratio_per_time_series | time_series_learning_ratio | learned_timestamps_ratio
                 models_series_lengths=None,  # training_time_series_lengths | time_series_training_lengths | learned_time_series_lengths | models_input_lengths
                 classifiers=None,
                 feature_extraction=True):
        """
        Parameters:
             base_classifier: Instance of a classifier to be cloned and learned for each timestamp to be classified.
                              Extreme gradient boosting (xgboost) is used by default.
            learned_timestamps_ratio: Proportion of timestamps the object will learn models on given any training set.
                              Attribute nb_classifiers is deduced from the time series' length in the training set.
               nb_classifiers: Number of classifiers in the list.
                              Attribute timestamps_ratio is deduced from the length of the training set.
            classifiers: Custom list of classifiers to be used (might be fitted already)
                              Attribute nb_classifiers is deduced to be the length of the custom_list.
                              Attribute timestamps_ratio is deduced from the time series' length in the training set.
        Attributes:
            self.classifiers: List of classifiers.
            self.models_series_length:
            self.
            self.base_classifier: Classifier to clone to get the classifiers in the list.
            self.timestamps_ratio: Proportion of timestamps to learn models on given any training set.
            self.nb_classifiers: Length of list.
            self.registered_timestamps

        """
        self.nb_classifiers = nb_classifiers
        self.base_classifier = base_classifier
        self.learned_timestamps_ratio = learned_timestamps_ratio
        self.models_series_lengths = models_series_lengths
        self.classifiers = classifiers
        self.feature_extraction = feature_extraction

    def __getitem__(self, item):
        return self.classifiers[item]

    def __len__(self):
        return self.nb_classifiers

    def get_params(self):
        return {
            "classifiers": self.classifiers,
            "models_series_lengths": self.models_series_lengths,
            "base_classifier": self.base_classifier,
            "nb_classifiers": self.nb_classifiers,
            "learned_timestamps_ratio": self.learned_timestamps_ratio,
            "feature_extraction": self.feature_extraction,
            "class_prior": self.class_prior,
            "max_series_length": self.max_series_length,
            "classes_": self.classes_
        }

    def fit(self, X, y, *args, **kwargs):
        """
        Takes as input a training set x of matrix shape (N, T) and its corresponding labels as a list y of N elements,
        where N is the number of time series and T their commune complete length. This method fits every classifier
        in the TimeIndexedClassifier object by historizing the training set into chronological subsets.
        Parameters:
            x: Training set of matrix shape (N, T) where:
                N is the number of time series
                T is the commune length of all complete time series
            y: List of the N corresponding labels of the training set.
        """
        # INPUT VALIDATION / INTEGRITY
        # time_series_learning_ratio compatibility
        if self.learned_timestamps_ratio is not None:
            if not isinstance(self.learned_timestamps_ratio, float) and not isinstance(self.learned_timestamps_ratio, int):
                raise TypeError(
                    "Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            if self.learned_timestamps_ratio <= 0 or self.learned_timestamps_ratio > 1:
                raise ValueError(
                    "Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            incompatible = []
            if self.nb_classifiers is not None:
                incompatible.append('nb_classifiers')
            if self.models_series_lengths is not None:
                incompatible.append('time_series_training_lengths')
            if self.classifiers is not None:
                incompatible.append('classifiers')
            if len(incompatible) > 0:
                raise ValueError(
                    f"Argument 'learned_timestamps_ratio' is not compatible with arguments {incompatible}.")

        # base_classifier and classifier_list compatibility
        if self.base_classifier is not None:
            if self.classifiers is not None:
                raise ValueError("Arguments 'base_classifier' and 'classifiers' are not compatible.")

        # Numerical coherence between arguments
        equal = []
        if self.nb_classifiers is not None:
            if not isinstance(self.nb_classifiers, int):
                raise TypeError("Argument 'nb_classifiers' should be a strictly positive int.")
            if self.nb_classifiers <= 0:
                raise ValueError("Argument 'nb_classifiers' should be a strictly positive int.")
            equal.append(('nb_classifiers', self.nb_classifiers))

        if self.classifiers is not None:
            if not isinstance(self.classifiers, list):
                raise TypeError("Argument 'classifiers' should be a list of classifier objects.")
            if len(self.classifiers) == 0:
                raise ValueError("List argument 'classifiers' is empty.")
            equal.append(('classifiers', len(self.classifiers)))

        if self.models_series_lengths is not None:
            if isinstance(self.models_series_lengths, list):
                self.models_series_lengths = np.array(self.models_series_lengths)
            elif not isinstance(self.models_series_lengths, np.ndarray):
                raise TypeError("Argument 'models_series_lengths' should be a list or array of positive int.")
            if len(self.models_series_lengths) == 0:
                raise ValueError("List argument 'models_series_lengths' is empty.")
            for t in self.models_series_lengths:
                if not isinstance(t, int):
                    raise TypeError("Argument 'models_series_lengths' should be a list or array of positive int.")
                if t < 0:
                    raise ValueError("Argument 'models_series_lengths' should be a list or array of positive int.")
            if len(np.unique(self.models_series_lengths)) != len(self.models_series_lengths):
                self.models_series_lengths = np.unique(self.models_series_lengths)
                warn("Removed duplicates timestamps in argument 'models_series_lengths'.")
            equal.append(('models_series_lengths', len(self.models_series_lengths)))

        if len(equal) >= 2:
            for i in range(len(equal) - 1):
                if equal[i][1] != equal[i + 1][1]:
                    raise ValueError(f"Contradictory values given to arguments {[e[0] for e in equal]}.")

        # feature_extraction check
        if not isinstance(self.feature_extraction, bool):
            raise TypeError("Argument 'feature_extraction' should be a bool.")

        # Check X and y
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a 2-dimensional list, array or DataFrame of size (N, T) "
                            "with N the number of examples and T the number of timestamps.")
        for n in range(len(X)):
            if len(X[n]) != len(X[0]):
                raise ValueError("All time series in the dataset should have the same length.")
        if len(X) == 0:
            raise ValueError("Dataset to fit 'X' is empty.")
        if X.ndim != 2:
            raise ValueError("X should be a 2-dimensional list, array or DataFrame of size (N, T) with N the number "
                             "of examples and T the number of timestamps.")
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be a list of labels of size N with N the number of examples in X")
        if len(y) != len(X):
            raise ValueError("y should be a list of classes of size N with N the number of examples in X")

        # ASSIGNMENTS
        # self.nb_classifiers
        self.max_series_length = X.shape[1]
        if self.nb_classifiers is None:
            if self.classifiers is not None:
                self.nb_classifiers = len(self.classifiers)
            elif self.models_series_lengths is not None:
                self.nb_classifiers = len(self.models_series_lengths)
            elif self.learned_timestamps_ratio is not None:
                self.nb_classifiers = int(self.learned_timestamps_ratio * self.max_series_length)
                if self.nb_classifiers == 0:
                    self.nb_classifiers = 1
            else:
                self.nb_classifiers = 20
                print("Using 'nb_classifiers=20 by default.")
        if self.nb_classifiers > self.max_series_length:
            if self.classifiers is not None or self.models_series_lengths is not None:
                raise ValueError(f"Not enough timestamps to learn {self.nb_classifiers} classifiers on")
            self.nb_classifiers = self.max_series_length
            warn(f"Not enough timestamps to learn {self.nb_classifiers} classifiers on. Changing number of classifiers "
                 f"to {self.max_series_length}.")

        # self.base_classifier
        if self.base_classifier is None:
            if self.classifiers is None:
                self.base_classifier = HistGradientBoostingClassifier()
                print("Using 'base_classifier = sklearn.ensemble.HistGradientBoostingClassifier() by default.")
        if self.models_series_lengths is None:
            self.models_series_lengths = np.array([int(self.max_series_length * (i + 1) / self.nb_classifiers) for i in
                                                   range(self.nb_classifiers)])
        else:
            self.models_series_lengths = np.sort(self.models_series_lengths)
        if self.classifiers is None:
            self.classifiers = [copy.deepcopy(self.base_classifier) for _ in range(self.nb_classifiers)]

        # Feature extraction and fitting
        for i, ts_length in enumerate(self.models_series_lengths):
            Xt = X[:, :ts_length]
            if self.feature_extraction:
                Xt = extract_features(Xt)
            self.classifiers[i].fit(Xt, y, *args, **kwargs)

        # Getting prior_probabilities
        try:
            self.classes_ = self.classifiers[0].classes_
            self.class_prior = np.array([np.count_nonzero(y == class_) / len(y) for class_ in self.classes_])
        except AttributeError:
            warn("Classifier does not have a 'classes_' attribute. Could not obtain prior probabilities.")
        return self

    def predict(self, X):
        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")
        if X.ndim != 2:
            raise ValueError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] not in self.models_series_lengths:
                for length in self.models_series_lengths[::-1]:
                    if length < time_series_lengths[i]:
                        X[i][length:] = np.nan
                        time_series_lengths[i] = length
                        truncated = True
                        break

        # Feature extraction is done on the whole predictions dataset to gain time.
        if self.feature_extraction is True:
            X = extract_features(X)
            inputs_lengths = get_time_series_lengths(X)
        else:
            inputs_lengths = time_series_lengths

        # Return prior if no classifier fitted for time series this short, predict with classifier otherwise
        predictions = []
        returned_priors = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] < self.models_series_lengths[0]:
                predictions.append(self.classes_[np.argmax(self.class_prior)])
                returned_priors = True
            else:
                predictions.append(
                    self.classifiers[np.nonzero(self.models_series_lengths == time_series_lengths[i])[0][0]]
                    .predict([time_series[:inputs_lengths[i]]])[0])

        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series could not be predicted since their length was below that of the training timestamps "
                 "of the first fitted classifier.")
        return np.array(predictions)

    def predict_proba(self, X):
        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] not in self.models_series_lengths:
                for length in self.models_series_lengths[::-1]:
                    if length < time_series_lengths[i]:
                        X[i][length:] = np.nan
                        time_series_lengths[i] = length
                        truncated = True
                        break

        # Feature extraction is done on the whole predictions dataset to gain time.
        if self.feature_extraction is True:
            X = extract_features(X)
            inputs_lengths = get_time_series_lengths(X)
        else:
            inputs_lengths = time_series_lengths

        # Return prior if no classifier fitted for time series this short, predict with classifier otherwise
        predictions = []
        returned_priors = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] < self.models_series_lengths[0]:
                predictions.append(self.class_prior)
                returned_priors = True
            else:
                predictions.append(
                    self.classifiers[np.nonzero(self.models_series_lengths == time_series_lengths[i])[0][0]]
                    .predict_proba([time_series[:inputs_lengths[i]]])[0])

        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series could not be predicted since their length was below that of the training timestamps "
                 "of the first fitted classifier.")
        return np.array(predictions)


class EarlyClassifier:
    """
    Combines a time-indexed array of classifiers with a trigger model to allow for simple early classification.
    Supports both predictions and triggering.
    May not support custom classifiers with unconventional methods names or structure.
    """
    def __init__(self,
                 misclassification_cost,
                 delay_cost,
                 nb_classifiers=None,
                 nb_intervals=5,
                 base_classifier=HistGradientBoostingClassifier(),
                 learned_timestamps_ratio=None,
                 feature_extraction=True,  # TODO: delete this once a better feature extraction is implemented
                 chronological_classifiers=None,
                 trigger_model=None):
        # models_series_lengths=None,
        # classifiers=None,
        # feature_extraction=True,
        # aggregation_function='max',
        self._misclassification_cost = misclassification_cost
        self._delay_cost = delay_cost
        self._nb_classifiers = nb_classifiers
        self._learned_timestamps_ratio = learned_timestamps_ratio
        self._base_classifier = base_classifier
        self._nb_intervals = nb_intervals
        self._feature_extraction = feature_extraction
        self.chronological_classifiers = chronological_classifiers
        self.trigger_model = trigger_model

    @property
    def misclassification_cost(self):
        return self.trigger_model.misclassification_cost

    @misclassification_cost.setter
    def misclassification_cost(self, value):
        self.misclassification_cost = value

    @property
    def delay_cost(self):
        return self.trigger_model.delay_cost

    @delay_cost.setter
    def delay_cost(self, value):
        self.delay_cost = value

    @property
    def nb_classifiers(self):
        return self.chronological_classifiers.nb_classifiers

    @nb_classifiers.setter
    def nb_classifiers(self, value):
        self.nb_classifiers = value

    @property
    def learned_timestamps_ratio(self):
        return self.chronological_classifiers.learned_timestamps_ratio

    @learned_timestamps_ratio.setter
    def learned_timestamps_ratio(self, value):
        self.learned_timestamps_ratio = value

    @property
    def base_classifier(self):
        return self.chronological_classifiers.base_classifier

    @base_classifier.setter
    def base_classifier(self, value):
        self.base_classifier = value

    @property
    def nb_intervals(self):
        return self.trigger_model.nb_intervals

    @nb_intervals.setter
    def nb_intervals(self, value):
        self.nb_intervals = value

    def get_params(self):
        return {
            "classifiers": self.chronological_classifiers.classifiers,
            "models_series_lengths": self.chronological_classifiers.models_series_lengths,
            "base_classifier": self.chronological_classifiers.base_classifier,
            "nb_classifiers": self.chronological_classifiers.nb_classifiers,
            "learned_timestamps_ratio": self.chronological_classifiers.learned_timestamps_ratio,
            "misclassification_cost": self.trigger_model.misclassification_cost,
            "delay_cost": self.trigger_model.delay_cost,
            "nb_intervals": self.trigger_model.nb_intervals,
            "aggregation_function": self.trigger_model.aggregation_function,
            "thresholds": self.trigger_model.thresholds,
            "transition_matrices": self.trigger_model.transition_matrices,
            "confusion_matrices": self.trigger_model.confusion_matrices,
            "feature_extraction": self.chronological_classifiers.feature_extraction,
            "class_prior": self.chronological_classifiers.class_prior,
            "max_series_length": self.chronological_classifiers.max_series_length,
            "classes_": self.chronological_classifiers.classes_,
            "multiclass": self.trigger_model.multiclass,
            "initial_cost": self.trigger_model.initial_cost,
        }

    def _fit_classifiers(self, X, y):
        self.chronological_classifiers.fit(X, y)
        return self

    def _fit_trigger_model(self, X, y):
        X_pred = np.stack([self.chronological_classifiers.predict_proba(X[:, :length])
                           for length in self.chronological_classifiers.models_series_lengths], axis=1)
        self.trigger_model.fit(X_pred, y, self.chronological_classifiers.classes_)

    def fit(self, X, y, val_proportion=.5):

        if not isinstance(X, list) and not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a 2-dimensional list, array or DataFrame of size (N, T) with N the number "
                            "of examples and T the number of timestamps.")

        val_index = int(len(X) * val_proportion)

        if self.chronological_classifiers is not None:
            if not isinstance(self.chronological_classifiers, ChronologicalClassifiers):
                raise ValueError(
                    "Argument 'chronological_classifiers' should be an instance of class 'ChronologicalClassifiers'.")
        else:
            self.chronological_classifiers = ChronologicalClassifiers(self._nb_classifiers,
                                                                      self._base_classifier,
                                                                      self._learned_timestamps_ratio,
                                                                      feature_extraction=self._feature_extraction)

        self._fit_classifiers(X[:val_index], y[:val_index])

        if self.trigger_model is not None:
            if not isinstance(self.trigger_model, EconomyGamma):
                raise ValueError(
                    "Argument 'trigger_model' should be an instance of class 'EconomyGamma'.")
        else:
            self.trigger_model = EconomyGamma(self._misclassification_cost, self._delay_cost,
                                              self.chronological_classifiers.models_series_lengths, self._nb_intervals)

        self._fit_trigger_model(X[val_index:], y[val_index:])
        # self.non_myopic = True if issubclass(type(self.trigger_model), NonMyopicTriggerModel) else False
        return self

    def predict(self, X):

        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.chronological_classifiers.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")
        if X.ndim != 2:
            raise ValueError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        classes = self.chronological_classifiers.predict(X)
        probas = self.chronological_classifiers.predict_proba(X)
        triggers, costs = self.trigger_model.predict(X, time_series_lengths)
        return classes, probas, triggers, costs


    """
    def predict(self, x):
        proba = self.classifiers.predict_proba()
        class_ = argmax(proba)
        if self.non_myopic:
            trigger, cost, forecasted_trigger, forecasted_costs = self.trigger_model.predict()
        else:
            trigger, cost = self.trigger_model.predict()
            forecasted_trigger, forecasted_costs = None, None
        return class_, proba, trigger, cost, forecasted_trigger, forecasted_costs
  
    def predict_class(self, x):
        class_ = self.classifiers.predict()
        return class_

    def predict_class_proba(self, x):
        proba = self.classifiers.predict_proba()
        return proba

    def predict_on_first_trigger(self, x):
        proba = self.classifiers.predict_proba()
        class_ = argmax(proba)
        if self.non_myopic:
            trigger_time, cost, forecasted_costs = self.trigger_model.predict_on_first_trigger()
        else:
            trigger_time, cost = self.trigger_model.predict_on_first_trigger()
            forecasted_costs = None, None
        return trigger_time, class_, proba, cost, forecasted_costs

    def predict_on_optimal_cost(self, x, y):
        proba = self.classifiers.predict_proba()
        class_ = argmax(proba)
        if self.non_myopic:
            trigger_time, cost, forecasted_costs = self.trigger_model.predict_on_first_trigger()
        else:
            trigger_time, cost = self.trigger_model.predict_on_first_trigger()
            forecasted_costs = None, None
        return trigger_time, class_, proba, cost, forecasted_costs
    """