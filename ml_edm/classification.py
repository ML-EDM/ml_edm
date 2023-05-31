import copy
from copy import deepcopy
import numpy as np
from pandas import DataFrame

from dataset import extract_features, get_time_series_lengths
from trigger_models import EconomyGamma

from sklearn.ensemble import GradientBoostingClassifier

from warnings import warn


class ChronologicalClassifiers:
    """
    A class used to manage a list of classifiers trained on time series of incrementing lengths following Scikit-Learn
    conventions.
    """

    # TODO : Comment on all functions
    # TODO : Add set_params
    # TODO : Multivariate data ??
    # TODO : Calibration
    # TODO : deal with NaN in dataset
    # TODO : "Serizalization of all the bazar" -> pickle dump and read
    # TODO : Make user feature extraction compatible with classifier selection
    # TODO : Optimize predict and better predict_proba output.
    # TODO : optimisation de seuil
    # TODO : Verbose, tqdm loadbar?
    # TODO : convert to numpy array
    # TODO : implement sparse matrix compatibility

    def __init__(self,
                 n_classifiers=None,
                 base_classifier=None,
                 learned_timestamps_ratio=None,  # learning_timestamps_ratio | learning_ratio_per_time_series | time_series_learning_ratio | learned_timestamps_ratio
                 classifiers_series_lengths=None,  # training_time_series_lengths | time_series_training_lengths | learned_time_series_lengths
                 classifiers=None,
                 feature_extraction=True):
        """
        Parameters:
             base_classifier: Instance of a classifier to be cloned and learned for each timestamp to be classified.
                              Extreme gradient boosting (xgboost) is used by default.
            learned_timestamps_ratio: Proportion of timestamps the object will learn models on given any training set.
                              Attribute n_classifiers is deduced from the time series' length in the training set.
               n_classifiers: Number of classifiers in the list.
                              Attribute timestamps_ratio is deduced from the length of the training set.
            classifiers: Custom list of classifiers to be used (might be fitted already)
                              Attribute n_classifiers is deduced to be the length of the custom_list.
                              Attribute timestamps_ratio is deduced from the time series' length in the training set.
        Attributes:
            self.classifiers: List of classifiers.
            self.classifiers_series_length:
            self.
            self.base_classifier: Classifier to clone to get the classifiers in the list.
            self.timestamps_ratio: Proportion of timestamps to learn models on given any training set.
            self.n_classifiers: Length of list.
            self.registered_timestamps

        """

        # INPUT VALIDATION / INTEGRITY
        # time_series_learning_ratio compatibility
        if learned_timestamps_ratio is not None:
            if not isinstance(learned_timestamps_ratio, float):
                raise TypeError("Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            if learned_timestamps_ratio <= 0 or learned_timestamps_ratio > 1:
                raise ValueError("Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            incompatible = []
            if n_classifiers is not None:
                incompatible.append('n_classifiers')
            if classifiers_series_lengths is not None:
                incompatible.append('time_series_training_lengths')
            if classifiers is not None:
                incompatible.append('classifiers')
            if len(incompatible) > 1:
                raise ValueError(f"Argument 'learned_timestamps_ratio' is not compatible with arguments {incompatible}.")

        # base_classifier and classifier_list compatibility
        if base_classifier is not None:
            if classifiers is not None:
                raise ValueError("Arguments 'base_classifier' and 'classifiers' are not compatible.")

        # Numerical coherence between arguments
        equal = []
        if n_classifiers is not None:
            if not isinstance(n_classifiers, int):
                raise TypeError("Argument 'n_classifiers' should be a strictly positive int.")
            if n_classifiers <= 0:
                raise ValueError("Argument 'n_classifiers' should be a strictly positive int.")
            equal.append(('n_classifiers', n_classifiers))

        if classifiers is not None:
            if not isinstance(classifiers, list):
                raise TypeError("Argument 'classifiers' should be a list of classifier objects.")
            if len(classifiers) == 0:
                raise ValueError("List argument 'classifiers' is empty.")
            equal.append(('classifiers', len(classifiers)))

        if classifiers_series_lengths is not None:
            if isinstance(classifiers_series_lengths, list):
                classifiers_series_lengths = np.array(classifiers_series_lengths)
            elif not isinstance(classifiers_series_lengths, np.ndarray):
                raise TypeError("Argument 'classifiers_series_lengths' should be a list or array of positive int.")
            if len(classifiers_series_lengths) == 0:
                raise ValueError("List argument 'classifiers_series_lengths' is empty.")
            for t in classifiers_series_lengths:
                if not isinstance(t, int):
                    raise TypeError("Argument 'classifiers_series_lengths' should be a list or array of positive int.")
                if t < 0:
                    raise ValueError("Argument 'classifiers_series_lengths' should be a list or array of positive int.")
            if len(np.unique(classifiers_series_lengths)) != len(classifiers_series_lengths):
                classifiers_series_lengths = np.unique(classifiers_series_lengths)
                warn("Removed duplicates timestamps in argument 'classifiers_series_lengths'.")
            equal.append(('classifiers_series_lengths', len(classifiers_series_lengths)))

        if len(equal) > 0:
            for i in range(len(equal) - 1):
                if equal[i][1] != equal[i+1][1]:
                    raise ValueError(f"Contradictory values given to arguments {[e[0] for e in equal]}.")

        # feature_extraction check
        if not isinstance(feature_extraction, bool):
            raise TypeError("Argument 'feature_extraction' should be a bool.")

        # ASSIGNMENTS
        # self.n_classifiers
        if n_classifiers is not None:
            self.n_classifiers = n_classifiers
        elif classifiers is not None:
            self.n_classifiers = len(classifiers)
        elif classifiers_series_lengths is not None:
            self.n_classifiers = len(classifiers_series_lengths)
        elif learned_timestamps_ratio is not None:
            self.n_classifiers = None
        else:
            self.n_classifiers = 20
            print("Using 'n_classifiers=20 by default.")

        # self.base_classifier
        if base_classifier is not None:
            self.base_classifier = base_classifier
        elif classifiers is not None:
            self.base_classifier = None
        else:
            self.base_classifier = GradientBoostingClassifier()
            print("Using 'base_classifier = sklearn.ensemble.GradientBoostingClassifier() by default.")

        # Other attributes (classifiers are initialized in fit)
        self.learned_timestamps_ratio = learned_timestamps_ratio if learned_timestamps_ratio is not None else None
        self.classifiers_series_lengths = np.sort(classifiers_series_lengths) if classifiers_series_lengths is not None else None
        self.classifiers = classifiers if classifiers is not None else None
        self.feature_extraction = feature_extraction
        self.class_prior = None
        self.max_series_length = None
        self.classes_ = None

    def __getitem__(self, item):
        return self.classifiers[item]

    def __len__(self):
        return self.n_classifiers

    def get_params(self):
        return {
            "classifiers": self.classifiers,
            "classifiers_series_lengths": self.classifiers_series_lengths,
            "base_classifier": self.base_classifier,
            "n_classifiers": self.n_classifiers,
            "learned_timestamps_ratio": self.learned_timestamps_ratio,
            "feature_extraction": self.feature_extraction,
            "prior": self.class_prior,
            "max_series_length": self.max_series_length,
            "classes": self.classes_
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
        # Check X and y
        X = copy.deepcopy(X)
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a two-dimensional list, array or DataFrame of size (N, T) "
                            "with N the number of examples and T the number of timestamps.")
        for n in range(len(X)):
            if len(X[n]) != len(X[0]):
                raise ValueError("All time series in the dataset should have the same length.")
        if len(X) == 0:
            warn("Dataset to predict 'X' is empty.")
        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(X, DataFrame):
            y = y.to_numpy()
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be a list of classes of size N with N the number of examples in X")
        if len(y) != len(X):
            raise ValueError("y should be a list of classes of size N with N the number of examples in X")

        # Data dependant attribute assignment and checks
        self.max_series_length = X.shape[1]
        if self.learned_timestamps_ratio is not None:
            self.n_classifiers = int(self.learned_timestamps_ratio * self.max_series_length)
            if self.n_classifiers == 0:
                self.n_classifiers = 1
        if self.max_series_length < self.n_classifiers:
            if self.classifiers is not None or self.classifiers_series_lengths is not None:
                raise ValueError(f"Not enough timestamps to learn {self.n_classifiers} classifiers on")
            self.n_classifiers = self.max_series_length
            warn(f"Not enough timestamps to learn {self.n_classifiers} classifiers on. Changing number of classifiers "
                 f"to {self.max_series_length}.")
        if self.classifiers is None:
            self.classifiers = [deepcopy(self.base_classifier) for _ in range(self.n_classifiers)]
        if self.classifiers_series_lengths is None:
            self.classifiers_series_lengths = np.array([int(self.max_series_length * (i + 1) / self.n_classifiers) for i in
                                                        range(self.n_classifiers)])
        # Feature extraction, fitting and calibration
        for i, ts_length in enumerate(self.classifiers_series_lengths):
            Xt = X[:, :ts_length]
            if self.feature_extraction:
                Xt = extract_features(Xt)
            self.classifiers[i].fit(Xt, y, *args, **kwargs)

        # Getting prior_probabilities
        # TODO: maybe make this try except?
        self.classes_ = self.classifiers[0].classes_
        self.class_prior = np.array([np.count_nonzero(y == class_) / len(y) for class_ in self.classes_])

        return self


    def predict(self, X):
        # TODO: check content of X? input validation with sklearn?
        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] not in self.classifiers_series_lengths:
                for length in self.classifiers_series_lengths[::-1]:
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
            if time_series_lengths[i] < self.classifiers_series_lengths[0]:
                predictions.append(self.classes_[np.argmax(self.class_prior)])
                returned_priors = True
            else:
                predictions.append(
                    self.classifiers[np.nonzero(self.classifiers_series_lengths == time_series_lengths[i])[0][0]]
                    .predict([time_series[:inputs_lengths[i]]])[0])

        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series could not be predicted since their length was below that of the training timestamps "
                 "of the first fitted classifier.")
        return np.array(predictions)

    def predict_proba(self, X):
        #TODO: check content of X? input validation with sklearn?
        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i, time_series in enumerate(X):
            if time_series_lengths[i] not in self.classifiers_series_lengths:
                for length in self.classifiers_series_lengths[::-1]:
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
            if time_series_lengths[i] < self.classifiers_series_lengths[0]:
                predictions.append(self.class_prior)
                returned_priors = True
            else:
                predictions.append(
                    self.classifiers[np.nonzero(self.classifiers_series_lengths == time_series_lengths[i])[0][0]]
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
                 n_classifiers=None,
                 base_classifier=None,
                 learned_timestamps_ratio=None,
                 classifiers_series_lengths=None,
                 classifiers=None,
                 feature_extraction=True,
                 chronological_classifiers=None,
                 misclassification_cost=None,
                 delay_cost=None,
                 nb_groups=0,
                 trigger_model_series_lengths=None,  # this should be at least contained in classifiers_series_lengths i think? what happens if trigger model exposed to time series of wrong size?
                 aggregation_type='gini',
                 trigger_model=None):

        self.chronological_classifiers = \
                ChronologicalClassifiers(n_classifiers, base_classifier, learned_timestamps_ratio, classifiers_series_lengths, classifiers, feature_extraction)\
                    if chronological_classifiers is None else chronological_classifiers
        self.trigger_model = EconomyGamma(misclassification_cost, delay_cost, nb_groups, trigger_model_series_lengths, aggregation_type) \
            if trigger_model is None else trigger_model
        #self.non_myopic = True if issubclass(type(self.trigger_model), NonMyopicTriggerModel) else False

    def fit(self, X, y, val_proportion=.5):
        val_index = int(len(X) * val_proportion)
        self.chronological_classifiers.fit(X[:val_index], y[:val_index])
        self.trigger_model.trigger_model_series_lengths = self.chronological_classifiers.classifiers_series_lengths
        X_pred = np.stack([self.chronological_classifiers.predict_proba(X[val_index:, :length])
                           for length in self.chronological_classifiers.classifiers_series_lengths], axis=1)
        self.trigger_model.fit(X_pred, y[val_index:], self.chronological_classifiers.classes_)
        return self

    def predict(self, X):
        # TODO: check content of X? input validation with sklearn?
        X = copy.deepcopy(X)
        # Validate X format
        if isinstance(X, list):
            padding = np.full((len(X), self.chronological_classifiers.max_series_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        elif isinstance(X, DataFrame):
            X = X.to_numpy
        elif not isinstance(X, np.ndarray):
            raise TypeError("X should be a list, array, or DataFrame of time series.")

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        classes = self.chronological_classifiers.predict(X)
        probas = self.chronological_classifiers.predict_proba(X)
        triggers, costs, forecasted_triggers, forecasted_costs = self.trigger_model.predict(X, time_series_lengths)
        return classes, probas, triggers, costs, forecasted_triggers, forecasted_costs


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