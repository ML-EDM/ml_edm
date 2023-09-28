import copy
import numpy as np
import pandas as pd

from deep_classifiers import DeepChronologicalClassifier
from dataset import extract_features, get_time_series_lengths
from trigger_models import *
from trigger_models_full import *
from utils import *

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from warnings import warn

# TODO : Multivariate data
# TODO : Change feature extraction
# TODO : Add calibration
# TODO : deal with NaN in dataset ? (Although HGBoost is nan compatible)
# TODO : Add set_params, setters etc... Check integrity issues doing so
# TODO : Optimize decision threshold
# TODO : Verbose, loadbar?
# TODO : implement sparse matrix compatibility
# TODO : Make classes skleearn estimators

def create_cost_matrices(
        timestamps, 
        misclassification_cost, 
        delay_cost=None, 
        alpha=1/4,
        cost_function=None):
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
    cost_matrices = np.array([[[(elem, alpha * delay_cost(t)) for elem in row] 
                               for row in misclassification_cost] for t in timestamps], dtype="i,f")
    
    if cost_function == 'sum':
        cost_matrices = np.array([[[sum(elem) for elem in row] 
                                   for row in matrix] for matrix in cost_matrices])

    return cost_matrices 

class ChronologicalClassifiers:
    """
    A class containing a list of classifiers to train on time series of incrementing lengths. Can be used along with a
    trigger model object such as an EconomyGamma instance to do tasks of early classification.

    Parameters:
        nb_classifiers: int, default=20
            Number of classifiers to be trained. If the number is inferior to the number of measures in the training
            time series, the models input lengths will be equally spaced from max_length/n_classifiers to max_length.
        base_classifier: classifier instance, default = sklearn.ensemble.HistGradientBoostingClassifier()
            Classifier instance to be cloned and trained for each input length.
        learned_timestamps_ratio: float, default=None
            Proportion of equally spaced time measurements/timestamps to use for training for all time series. A float
            between 0 and 1. Incompatible with parameters 'nb_classifiers', 'models_input_lengths' and 'classifiers'.
        models_input_lengths: numpy.ndarray, default = None
            Array containing the numbers of time measurements/input length that each classifier is trained on.
            Argument 'nb_classifiers' is deduced from the length of this list.
        classifiers: numpy.ndarray, default = None
            List or array containing the classifier instances to be trained. Argument 'nb_classifiers' is deduced from
            the length of this list.
        feature_extraction: boolean, default=False TODO: Update this once a better feature extraction is implemented.
            Boolean indicating whether to use the default feature extraction. Currently implemented feature extraction
            is quite slow, wordy, does not work with time series with less than 12 measurements and does not seem very
            effective. We do not recommend its use without experimentation.
    Attributes:
        class_prior: numpy.ndarray
            Class prior probabilities vector obtained from the training set. The order of the classes is detailed in the
            'classes_' argument
        classes_:
            Array containing the name of each classes ordered the same way as the 'class_prior' argument. Obtained from
            the 'classes_' argument of the first classifier in the 'classifiers' list of the ChronologicalClassifiers
            object.
        max_series_length: int
            Maximum number of measurements contained by a time series in the training set.
    """

    def __init__(self,
                 nb_classifiers=None,
                 base_classifier=None,
                 learned_timestamps_ratio=None,
                 models_input_lengths=None,
                 classifiers=None,
                 min_length=1,
                 feature_extraction=False, # TODO: make this True once a better feature extraction is implemented.
                 random_state=44):  
        
        self.nb_classifiers = nb_classifiers
        self.base_classifier = base_classifier
        self.learned_timestamps_ratio = learned_timestamps_ratio
        self.models_input_lengths = models_input_lengths

        self.min_length = min_length
        self.classifiers = classifiers

        self.feature_extraction = feature_extraction
        self.random_state = random_state

        self.max_length = None
        self.classes_ = None
        self.class_prior = None

    def __getitem__(self, item):
        return self.classifiers[item]

    def __len__(self):
        return self.nb_classifiers

    def get_params(self):
        return {
            "classifiers": self.classifiers,
            "models_input_lengths": self.models_input_lengths,
            "base_classifier": self.base_classifier,
            "nb_classifiers": self.nb_classifiers,
            "learned_timestamps_ratio": self.learned_timestamps_ratio,
            "min_length": self.min_length,
            "feature_extraction": self.feature_extraction,
            "class_prior": self.class_prior,
            "classes_": self.classes_,
            "max_series_length": self.max_series_length,
        }

    def fit(self, X, y, *args, **kwargs):
        """
        This method fits every classifier in the ChronologicalClassifiers object by truncating the time series of the
        training set to the input lengths contained in the attribute models_input_lengths'. The prior probabilities are
        also saved.

        Parameters:
            X: np.ndarray
                Training set of matrix shape (N, T) where:
                    N is the number of time series
                    T is the commune length of all complete time series
            y: nd.ndarray
                List of the N corresponding labels of the training set.
        """
        # INPUT VALIDATION / INTEGRITY
        # time_series_learning_ratio compatibility
        if self.learned_timestamps_ratio is not None:
            if not isinstance(self.learned_timestamps_ratio, float) \
                    and not isinstance(self.learned_timestamps_ratio, int):
                raise TypeError(
                    "Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            
            if self.learned_timestamps_ratio <= 0 or self.learned_timestamps_ratio > 1:
                raise ValueError(
                    "Argument 'learned_timestamps_ratio' should be a strictly positive float between 0 and 1.")
            
            incompatible = []
            if self.nb_classifiers is not None:
                incompatible.append('nb_classifiers')
            if self.models_input_lengths is not None:
                incompatible.append('models_input_lengths')
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

        if self.models_input_lengths is not None:
            if isinstance(self.models_input_lengths, list):
                self.models_input_lengths = np.array(self.models_input_lengths)
            elif not isinstance(self.models_input_lengths, np.ndarray):
                raise TypeError("Argument 'models_input_lengths' should be a list or array of positive int.")
            if len(self.models_input_lengths) == 0:
                raise ValueError("List argument 'models_input_lengths' is empty.")
            for t in self.models_input_lengths:
                if not (isinstance(t, np.int32) or isinstance(t, np.int64)):
                    raise TypeError("Argument 'models_input_lengths' should be a list or array of positive int.")
                if t < 0:
                    raise ValueError("Argument 'models_input_lengths' should be a list or array of positive int.")
            if len(np.unique(self.models_input_lengths)) != len(self.models_input_lengths):
                self.models_input_lengths = np.unique(self.models_input_lengths)
                warn("Removed duplicates timestamps in argument 'models_input_lengths'.")
            equal.append(('models_input_lengths', len(self.models_input_lengths)))

        if len(equal) >= 2:
            for i in range(len(equal) - 1):
                if equal[i][1] != equal[i + 1][1]:
                    raise ValueError(f"Contradictory values given to arguments {[e[0] for e in equal]}.")

        # feature_extraction check
        if not isinstance(self.feature_extraction, bool):
            raise TypeError("Argument 'feature_extraction' should be a bool.")

        # Check X and y
        X, y = check_X_y(X, y)

        # ASSIGNMENTS
        # self.nb_classifiers
        self.max_length = X.shape[1]
        if self.nb_classifiers is None:
            if self.classifiers is not None:
                self.nb_classifiers = len(self.classifiers)
            elif self.models_input_lengths is not None:
                self.nb_classifiers = len(self.models_input_lengths)
            elif self.learned_timestamps_ratio is not None:
                self.nb_classifiers = int(self.learned_timestamps_ratio * self.max_length)
                if self.nb_classifiers == 0:
                    self.nb_classifiers = 1
            else:
                self.nb_classifiers = 20
                print("Using 'nb_classifiers=20 by default.")

        if self.nb_classifiers > self.max_length:
            if self.classifiers is not None or self.models_input_lengths is not None:
                raise ValueError(f"Not enough timestamps to learn {self.nb_classifiers} classifiers on")
            
            self.nb_classifiers = self.max_length
            warn(f"Not enough timestamps to learn {self.nb_classifiers} classifiers on."
                 f"Changing number of classifiers to {self.max_length}.")

        # self.base_classifier
        if self.base_classifier is None:
            if self.classifiers is None:
                self.base_classifier = HistGradientBoostingClassifier(random_state=self.random_state)
                print("Using 'base_classifier = HistGradientBoostingClassifier() by default.")

        if self.models_input_lengths is None:
            self.models_input_lengths = list(set(
                [int((self.max_length - self.min_length) * i / self.nb_classifiers) + self.min_length
                 for i in range(1, self.nb_classifiers+1)]
            ))
            self.nb_classifiers = len(self.models_input_lengths)

        self.models_input_lengths = np.sort(self.models_input_lengths)

        if self.classifiers is None:
            self.classifiers = [copy.deepcopy(self.base_classifier) for _ in range(self.nb_classifiers)]

        # FEATURE EXTRACTION AND FITTING
        for i, ts_length in enumerate(self.models_input_lengths):
            Xt = X[:, :ts_length]
            if self.feature_extraction:
                Xt = extract_features(Xt)         
            self.classifiers[i].fit(Xt, y, *args, **kwargs)

        # GETTING PRIOR PROBABILITIES
        try:
            self.classes_ = self.classifiers[0].classes_ 
            self.class_prior = np.array([np.sum(y == class_) / len(y) for class_ in self.classes_])
        except AttributeError:
            warn("Classifier does not have a 'classes_' attribute. Could not obtain prior probabilities.")

        return self

    def predict(self, X):
        """
        Predict a dataset of time series of various lengths using the right classifier in the ChronologicalClassifiers
        object. If a time series has a different number of measurements than the values in 'models_input_lengths', the
        time series is truncated to the closest compatible length. If its length is shorter than the first length in
        'models_input_lengths', the prior probabilities are used. Returns the most probable class of each series.
        Parameters:
            X: np.ndarray
            Dataset of time series of various sizes to predict. An array of size (N*max_T) where N is the number of
            time series, max_T the max number of measurements in a time series and where empty values are filled with
            nan. Can also be a pandas DataFrame or a list of lists.
        Returns:
            np.ndarray containing the classifier predicted class for each time series in the dataset.
        """
        # Validate X format
        X, _ = check_X_y(X, None, equal_length=False)

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i in range(len(X)):
            if time_series_lengths[i] not in self.models_input_lengths:
                for length in self.models_input_lengths[::-1]:
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
        for i, x in enumerate(X):
            if time_series_lengths[i] < self.models_input_lengths[0]:
                predictions.append(self.classes_[np.argmax(self.class_prior)])
                returned_priors = True
            else:
                clf_idx = np.where(self.models_input_lengths == time_series_lengths[i])
                predictions.append(
                    self.classifiers[clf_idx[0][0]].predict(x[None, :inputs_lengths[i]])[0]
                )

        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return np.array(predictions)

    def predict_proba(self, X, past_probas=False):
        """
        Predict a dataset of time series of various lengths using the right classifier in the ChronologicalClassifiers
        object. If a time series has a different number of measurements than the values in 'models_input_lengths', the
        time series is truncated to the closest compatible length. If its length is shorter than the first length in
        'models_input_lengths', the prior probabilities are used. Returns the class probabilities vector of each series.
        Parameters:
            X: np.ndarray
            Dataset of time series of various sizes to predict. An array of size (N*max_T) where N is the number of
            time series, max_T the max number of measurements in a time series and where empty values are filled with
            nan. Can also be a pandas DataFrame or a list of lists.
        Returns:
            np.ndarray containing the classifier class probabilities array for each time series in the dataset.
        """
        # Validate X format with varying series lengths
        X, _ = check_X_y(X, None, equal_length=False)

        # Get time series lengths
        time_series_lengths = get_time_series_lengths(X)

        # Truncate time series to classifier compatible length
        truncated = False
        for i in range(len(X)):
            if time_series_lengths[i] not in self.models_input_lengths:
                for length in self.models_input_lengths[::-1]:
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

        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for i, x in enumerate(X):
            if time_series_lengths[i] < self.models_input_lengths[0]:
                predictions.append(self.class_prior)
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.models_input_lengths == time_series_lengths[i]
                )
                if not past_probas:
                    predictions.append(
                        self.classifiers[clf_idx[0][0]].predict_proba(x[None, :inputs_lengths[i]])[0]
                    )
                else:
                    if time_series_lengths[i] != self.models_input_lengths[0]:
                        all_probas = [
                            self.classifiers[j].predict_proba(x[None, :self.models_input_lengths[j]])[0]
                            for j in range(clf_idx[0][0]+1)
                        ]
                    else:
                        all_probas = self.classifiers[0].predict_proba(x[None, :inputs_lengths[i]])[0]
                      
                    predictions.append(all_probas)   

        if not past_probas:
            predictions = np.array(predictions)
        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return predictions


class EarlyClassifier:
    """
    Objects that can predict the class of an incomplete time series as well as the best time to trigger the decision as
    a time series is revealed to allow for early classification tasks.
    Combines a ChronologicalClassifier instance and a trigger model (EconomyGamma instance by default). These objects
    can be used separately if a more technical use is needed.

    Parameters:
        misclassification_cost: numpy.ndarray
            Array of size Y*Y where Y is the number of classes and where each value at indices
            [i,j] represents the cost of predicting class j when the actual class is i. Usually, diagonals of the
            matrix are all zeros. This cost must be defined by a domain expert and be expressed in the same unit
            as the delay cost.
        delay_cost: python function
            Function that takes as input a time series input length and returns the timely cost of waiting
            to obtain such number of measurements given the task. This cost must be defined by a domain expert and
            be expressed in the same unit as the misclassification cost.
        nb_classifiers: int, default=20
            Number of classifiers to be trained. If the number is inferior to the number of measures in the training
            time series, the models input lengths will be equally spaced from max_length/n_classifiers to max_length.
        nb_intervals: int, default=5
            Number of groups to aggregate the training time series into for each input length during learning of the
            trigger model. The optimal value of this hyperparameter may depend on the task.
        base_classifier: classifier instance, default = sklearn.ensemble.HistGradientBoostingClassifier()
                Classifier instance to be cloned and trained for each input length.
        learned_timestamps_ratio: float, default=None
            Proportion of equally spaced time measurements/timestamps to use for training for all time series. A float
            between 0 and 1. Incompatible with parameters 'nb_classifiers'
        chronological_classifiers: ChronologicalClassifier()
            Custom instance of the ChronologicalClassifier object used in combination with the trigger model.
        trigger_model: EconomyGamma()
            Custom instance of the EconomyGamma() object used in combination with the chronological classifiers.

    Attributes:
        All attributes of the instance can be accessed from their 'chronological_classifiers' and 'trigger_model'
        objects.
        Check documentation of ChronologicalClassifiers and EconomyGamma for more detail on these attributes.
    """
    def __init__(self,
                 misclassification_cost,
                 delay_cost,
                 nb_classifiers=None,
                 nb_intervals=5,
                 base_classifier=HistGradientBoostingClassifier(),
                 learned_timestamps_ratio=None,
                 min_length=1,
                 chronological_classifiers=None,
                 trigger_model=None,
                 random_state=44):
        
        self.misclassification_cost = misclassification_cost
        self.delay_cost = delay_cost

        self._nb_classifiers = nb_classifiers
        self._learned_timestamps_ratio = learned_timestamps_ratio
        self._base_classifier = base_classifier

        self._min_length = min_length
        self._nb_intervals = nb_intervals

        self.chronological_classifiers = chronological_classifiers
        self.trigger_model = trigger_model
        self.random_state = random_state

    # Properties are used to give direct access to the 
    # chronological classifiers and trigger models arguments.
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
    
    @property
    def min_length(self):
        return self.chronological_classifiers.min_length

    @min_length.setter
    def min_length(self, value):
        self.min_length = value

    def get_params(self):
        return {
            "classifiers": self.chronological_classifiers.classifiers,
            "models_input_lengths": self.chronological_classifiers.models_input_lengths,
            "base_classifier": self.chronological_classifiers.base_classifier,
            "nb_classifiers": self.chronological_classifiers.nb_classifiers,
            "learned_timestamps_ratio": self.chronological_classifiers.learned_timestamps_ratio,
            "misclassification_cost": self.misclassification_cost,
            "delay_cost": self.delay_cost,
            "cost_matrices": self.trigger_model.cost_matrices,
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
                           for length in self.chronological_classifiers.models_input_lengths], axis=1)
        self.trigger_model.fit(X, X_pred, y)
        #self.trigger_model.fit(X, y)

    def fit(self, X, y, val_proportion=.7):
        """
        Parameters:
            X: np.ndarray
                Training set of matrix shape (N, T) where:
                N is the number of time series
                T is the commune length of all complete time series
            y: nd.ndarray
                List of the N corresponding labels of the training set.
            val_proportion: float
                Value between 0 and 1 representing the proportion of data to be used by the trigger_model for training.
        """

        # DEFINE THE SEPARATION INDEX FOR VALIDATION DATA
        if val_proportion != 0:
            X_clf, X_trigger, y_clf, y_trigger = train_test_split(
                X, y, test_size=(1-val_proportion), random_state=self.random_state
            )
        else:
            X_clf = X_trigger = X
            y_clf = y_trigger = y
        
        # FIT CLASSIFIERS
        if self.chronological_classifiers is not None:
            if not isinstance(self.chronological_classifiers, ChronologicalClassifiers):
                raise ValueError(
                    "Argument 'chronological_classifiers' should be an instance of class 'ChronologicalClassifiers'.")
        else:
            self.chronological_classifiers = ChronologicalClassifiers(self._nb_classifiers,
                                                                      self._base_classifier,
                                                                      self._learned_timestamps_ratio,
                                                                      min_length=self._min_length)
            #self.chronological_classifiers = DeepChronologicalClassifier()

        self._fit_classifiers(X_clf, y_clf)

        # OBTAIN COST MATRICES AND FIT TRIGGER MODEL
        if self.trigger_model is not None:
            if not isinstance(self.trigger_model, EconomyGamma):
                raise ValueError(
                    "Argument 'trigger_model' should be an instance of class 'EconomyGamma'.")
        else:
            self.cost_matrices = create_cost_matrices(self.chronological_classifiers.models_input_lengths,
                                                 self.misclassification_cost, self.delay_cost, cost_function="sum")
            #self.trigger_model = EconomyGamma(self.cost_matrices, self.chronological_classifiers.models_input_lengths,
            #                                  self._nb_intervals)
            #self.trigger_model = StoppingRule(self.cost_matrices, self.chronological_classifiers.models_input_lengths, 
            #                                  stopping_rule="SR1", n_jobs=2)
            #self.trigger_model = TEASER(self.cost_matrices, self.chronological_classifiers.models_input_lengths, 
            #                            objective='avg_cost', n_jobs=1)
            #self.trigger_model = ECEC(self.cost_matrices, self.chronological_classifiers.models_input_lengths, n_jobs=2)
            #self.trigger_model = ProbabilityThreshold(self.cost_matrices, self.chronological_classifiers.models_input_lengths, n_jobs=1)
            #self.trigger_model = ECDIRE(self.chronological_classifiers, n_jobs=2)
            #self.trigger_model = EDSC(min_length=5, max_length=12, n_jobs=3)
            #self.trigger_model = ECTS(self.chronological_classifiers.models_input_lengths, support=0, relaxed=False, n_jobs=1)
            self.trigger_model = CALIMERA(self.cost_matrices, self.chronological_classifiers.models_input_lengths, alpha=1/4)

        self._fit_trigger_model(X_trigger, y_trigger)
        # self.chronological_classifiers = self.trigger_model.chronological_classifiers # if ECDIRE
        # self.non_myopic = True if issubclass(type(self.trigger_model), NonMyopicTriggerModel) else False
        return self

    def predict(self, X):
        """
        Predicts the class, class probabilities vectors, trigger indication and expected costs of the time series
        contained in X.
        Parameters:
            X: np.ndarray
                Dataset of time series of various sizes to predict. An array of size (N*max_T) where N is the number of
                time series, max_T the max number of measurements in a time series and where empty values are filled
                with nan. Can also be a pandas DataFrame or a list of lists.
        Returns:
            classes: np.ndarray:
                Array containing the predicted class of each time series in X.
            probas: np.ndarray
                Array containing the predicted class probabilities vectors of each time series in X.
            triggers: np.ndarray
                Array of booleans indicating whether to trigger the decision immediately with the current prediction
                (True) or to wait for more data (False) for each time series in X.
            costs: np.ndarray
                Array of arrays containing the expected costs of waiting longer for each next considered measurement for
                each time series in X. Using argmin on each row outputs the number of timestamps to wait for until the
                next expected trigger indication.
        """

        # Validate X format
        X, _ = check_X_y(X, None, equal_length=False)

        # Predict
        classes = self.chronological_classifiers.predict(X)
        probas = self.chronological_classifiers.predict_proba(X, past_probas=False)
        triggers, costs = self.trigger_model.predict(X, probas)
        #classes, triggers = self.trigger_model.predict(X)
        #costs = None

        return classes, probas, triggers, costs
