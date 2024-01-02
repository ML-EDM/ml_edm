import copy
import numpy as np
import pandas as pd

from deep.deep_classifiers import DeepChronologicalClassifier
from dataset import extract_features, get_time_series_lengths
from features_extraction import Feature_extractor
from cost_matrice import CostMatrices
from trigger_models import *
from trigger_models_full import *
from utils import *
from metrics import average_cost

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.calibration import CalibratedClassifierCV

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
        alpha=1/2,
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
                 base_classifier=None,
                 sampling_ratio=None,
                 models_input_lengths=None,
                 classifiers=None,
                 min_length=1,
                 feature_extraction=None, # TODO: make this True once a better feature extraction is implemented.
                 calibration=True,
                 random_state=44):  
        
        self.base_classifier = base_classifier
        self.sampling_ratio = sampling_ratio
        self.models_input_lengths = models_input_lengths

        self.classifiers = classifiers

        self.min_length = min_length
        self.feature_extraction = feature_extraction
        self.calibration = calibration
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
            "sampling_ratio": self.sampling_ratio,
            "min_length": self.min_length,
            "feature_extraction": self.feature_extraction,
            "class_prior": self.class_prior,
            "classes_": self.classes_,
            "max_series_length": self.max_length,
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
        self.max_length = X.shape[1]

        if self.sampling_ratio is not None:
            if not isinstance(self.sampling_ratio, float) \
                    and not isinstance(self.sampling_ratio, int):
                raise TypeError(
                    "Argument 'sampling_ratio' should be a strictly positive float between 0 and 1.")
            
            if self.sampling_ratio <= 0 or self.sampling_ratio > 1:
                raise ValueError(
                    "Argument 'sampling_ratio' should be a strictly positive float between 0 and 1.")
            
            incompatible = []
            if self.models_input_lengths is not None:
                incompatible.append('models_input_lengths')
            if self.classifiers is not None:
                incompatible.append('classifiers')
            if len(incompatible) > 0:
                raise ValueError(
                    f"Argument 'learned_timestamps_ratio' is not compatible with arguments {incompatible}.")
        else:
            self.sampling_ratio = 0.05
            warn("No sampling ratio provided, using default 5'%' sampling")
            
        self.nb_classifiers = np.minimum(int(1/self.sampling_ratio), self.max_length - self.min_length + 1)

        # base_classifier and classifier_list compatibility
        if self.base_classifier is not None:
            if self.classifiers is not None:
                raise ValueError("Arguments 'base_classifier' and 'classifiers' are not compatible.")
        else:
            if self.classifiers is None:
                self.base_classifier = HistGradientBoostingClassifier(random_state=self.random_state)
                warn("Using 'base_classifier = HistGradientBoostingClassifier() by default.")
        
        if self.classifiers is None:
            self.classifiers = [copy.deepcopy(self.base_classifier) for _ in range(self.nb_classifiers)]

        # Numerical coherence between arguments
        equal = []
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
        else:
            self.models_input_lengths = np.array(list(set(
                [int((self.max_length - self.min_length) * i / self.nb_classifiers) + self.min_length
                 for i in range(1, self.nb_classifiers+1)]
            )))
            self.nb_classifiers = len(self.models_input_lengths)
        self.models_input_lengths = np.sort(self.models_input_lengths)

        if len(equal) >= 2:
            for i in range(len(equal) - 1):
                if equal[i][1] != equal[i + 1][1]:
                    raise ValueError(f"Contradictory values given to arguments {[e[0] for e in equal]}.")

        # feature_extraction check
        """
        if not isinstance(self.feature_extraction, str):
            raise TypeError("Argument 'feature_extraction' should be a string.")
        """

        # Check X and y
        X, y = check_X_y(X, y)

        # FEATURE EXTRACTION AND FITTING
        self.extractors = []
        for i, ts_length in enumerate(self.models_input_lengths):
            Xt = X[:, :ts_length]
            if self.feature_extraction:
                scale = True if self.feature_extraction['method'] == 'minirocket' else False
                self.extractors.append(Feature_extractor(self.feature_extraction['method'], scale, 
                                                         kwargs=self.feature_extraction['params']).fit(Xt, y))
                Xt = self.extractors[-1].transform(Xt)

            if self.calibration:
                Xt_clf, X_calib, y_clf, y_calib = train_test_split(Xt.reshape(len(X),-1), y, test_size=0.3, stratify=y,
                                                                    random_state=self.random_state)
            else:
                Xt_clf, y_clf = (Xt.reshape(len(X),-1), y)
            self.classifiers[i].fit(Xt_clf, y_clf, *args, **kwargs)
            
            if self.calibration:
                calib_clf = CalibratedClassifierCV(self.classifiers[i], cv='prefit')
                self.classifiers[i] = calib_clf.fit(X_calib, y_calib)

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

        # Group X by batch of same length
        truncated = False
        grouped_X = {}
        for serie in X:
            length = len(serie)
            if length not in self.models_input_lengths and \
                length > self.models_input_lengths[0]:
                # truncate to nearest valid timestamp
                filtered = filter(lambda x: x <= length, self.models_input_lengths)
                length = min(filtered, key=lambda x: length-x, default=None)
                if length:
                    serie = serie[:length]
                    truncated = True

            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]

        # Return prior if no classifier fitted for time series this short, predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.models_input_lengths[0]:
                preds_prior = [self.classes_[np.argmax(self.class_prior)]
                               for _ in range(len(series))]
                predictions.append(np.array(preds_prior))
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.models_input_lengths == length
                )[0][0]
                series = np.array(series)
                if self.feature_extraction:
                    series = self.extractors[clf_idx].transform(series)
                predictions.append(
                    self.classifiers[clf_idx].predict(series)
                )  

        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return np.vstack(predictions).squeeze()

    def predict_proba(self, X):
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

        # Group X by batch of same length
        truncated = False
        grouped_X = {}
        for serie in X:
            length = len(serie)
            if length not in self.models_input_lengths and \
                length > self.models_input_lengths[0]:
                # truncate to nearest valid timestamp
                filtered = filter(lambda x: x <= length, self.models_input_lengths)
                length = min(filtered, key=lambda x: length-x, default=None)
                if length:
                    serie = serie[:length]
                    truncated = True

            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]

        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.models_input_lengths[0]:
                predictions.append(np.ones((len(series), len(self.class_prior))) * self.class_prior)
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.models_input_lengths == length
                )[0][0]
                series = np.array(series).reshape(len(series),-1)
                if self.feature_extraction and os.path.isdir(str(self.feature_extraction)):
                    fts_idx = clf_idx
                    if hasattr(self, "prev_models_input_lengths"):
                        fts_idx = np.where(self.prev_models_input_lengths == length)[0][0]
                    series = np.load(self.feature_extraction+f"/features_{fts_idx}.npy")
                elif self.feature_extraction:
                    series = self.extractors[clf_idx].transform(series)

                predictions.append(
                    self.classifiers[clf_idx].predict_proba(series)
                )  
        # Send warnings if necessary
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return np.vstack(predictions)
    
    def predict_past_proba(self, X):
        # Validate X format with varying series lengths
        X, _ = check_X_y(X, None, equal_length=False)

        # Group X by batch of same length
        truncated = False
        grouped_X = {}
        for serie in X:
            length = len(serie)
            if length not in self.models_input_lengths and \
                length > self.models_input_lengths[0]:
                # truncate to nearest valid timestamp
                filtered = filter(lambda x: x <= length, self.models_input_lengths)
                length = min(filtered, key=lambda x: length-x, default=None)
                if length:
                    serie = serie[:length]
                    truncated = True

            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]

        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.models_input_lengths[0]:
                priors = np.ones((len(series), len(self.class_prior))) * self.class_prior
                predictions.append(priors)
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.models_input_lengths == length
                )[0][0]
                if length != self.models_input_lengths[0]:
                    series = np.array(series) # allow for slicing
                    
                    if self.feature_extraction and os.path.isdir(str(self.feature_extraction)):
                        partial_series = [np.load(self.feature_extraction+f"/features_{j}.npy") 
                                          for j in range(clf_idx+1)]
                    elif self.feature_extraction:
                        partial_series = [
                            self.extractors[j].transform(
                            series[:, :self.models_input_lengths[j]])
                            for j in range(clf_idx+1)
                        ]
                    else:
                        partial_series = [series[:, :self.models_input_lengths[j]].reshape(len(series),-1)
                                          for j in range(clf_idx+1)]

                    all_probas = [
                        self.classifiers[j].predict_proba(x) for j, x in enumerate(partial_series)
                    ]
                    all_probas = list(
                        np.array(all_probas).transpose((1,0,2))
                    )
                else:
                    if self.feature_extraction:
                        series = self.extractors[0].transform(series)

                    all_probas = self.classifiers[0].predict_proba(np.array(series))
                    all_probas = list(np.expand_dims(all_probas, 1))

                predictions.extend(all_probas)

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
                 chronological_classifiers=None,
                 prefit_classifiers=False, 
                 trigger_model=None,
                 trigger_params={},
                 cost_matrices=None,
                 random_state=44):
        
        self.cost_matrices = cost_matrices
        self.chronological_classifiers = copy.deepcopy(chronological_classifiers)
        self.prefit = prefit_classifiers

        self.trigger_model = trigger_model
        self.trigger_params = trigger_params

        self.random_state = random_state

    # Properties are used to give direct access to the 
    # chronological classifiers and trigger models arguments.
    @property
    def models_input_lengths(self):
        return self.chronological_classifiers.models_input_lengths
    
    @property
    def nb_classifiers(self):
        return self.chronological_classifiers.nb_classifiers

    @nb_classifiers.setter
    def nb_classifiers(self, value):
        self.nb_classifiers = value

    @property
    def sampling_ratio(self):
        return self.chronological_classifiers.sampling_ratio

    @sampling_ratio.setter
    def sampling_ratio(self, value):
        self.sampling_ratio = value

    @property
    def base_classifier(self):
        return self.chronological_classifiers.base_classifier

    @base_classifier.setter
    def base_classifier(self, value):
        self.base_classifier = value
        return self.trigger_model.nb_intervals
    
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
            "sampling_ratio": self.chronological_classifiers.sampling_ratio,
            "cost_matrices": self.trigger_model.cost_matrices,
            "aggregation_function": self.trigger_model.aggregation_function,
            "thresholds": self.trigger_model.thresholds,
            "transition_matrices": self.trigger_model.transition_matrices,
            "confusion_matrices": self.trigger_model.confusion_matrices,
            "feature_extraction": self.chronological_classifiers.feature_extraction,
            "class_prior": self.chronological_classifiers.class_prior,
            "max_length": self.chronological_classifiers.max_length,
            "classes_": self.chronological_classifiers.classes_,
            "multiclass": self.trigger_model.multiclass,
            "initial_cost": self.trigger_model.initial_cost,
        }

    def _fit_classifiers(self, X, y):
        self.chronological_classifiers.fit(X, y)
        return self

    def _fit_trigger_model(self, X, y):
        #X_pred = np.stack([self.chronological_classifiers.predict_proba(X[:, :length])
        #                   for length in self.chronological_classifiers.models_input_lengths], axis=1)
        
        if self.trigger_model.require_classifiers:
            X_probas = np.stack(self.chronological_classifiers.predict_past_proba(X))
            self.trigger_model.fit(X, X_probas, y)
        else:
            self.trigger_model.fit(X, y)

    def fit(self, X, y, trigger_proportion=0.3):
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
        if trigger_proportion != 0:
            X_clf, X_trigger, y_clf, y_trigger = train_test_split(
                X, y, test_size=trigger_proportion, random_state=self.random_state
            )
        else:
            X_clf = X_trigger = X
            y_clf = y_trigger = y
        
        # FIT CLASSIFIERS
        if self.chronological_classifiers is not None:
            if not isinstance(self.chronological_classifiers, ChronologicalClassifiers) and \
                not isinstance(self.chronological_classifiers, DeepChronologicalClassifier):
                raise ValueError(
                    "Argument 'chronological_classifiers' should be an instance of class 'ChronologicalClassifiers'.")
        else:
            self.chronological_classifiers = ChronologicalClassifiers(
                base_classifier=HistGradientBoostingClassifier(),
                learned_timestamps_ratio=0.05,
                min_length=1
            )
            #self.chronological_classifiers = DeepChronologicalClassifier()
        if not self.prefit:
            self._fit_classifiers(X_clf, y_clf)

        # FIT TRIGGER MODEL
        if not self.cost_matrices:
            self.cost_matrices = CostMatrices(timestamps=self.models_input_lengths, 
                                              n_classes=len(np.unique(y)), 
                                              alpha=1.)
            warn("No cost matrices defined, using alpha = 1 by default")

        if self.trigger_model is not None:
            if self.trigger_model == "economy":
                self.trigger_model = EconomyGamma(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "proba_threshold":
                self.trigger_model = ProbabilityThreshold(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "stopping_rule":
                self.trigger_model = StoppingRule(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "teaser":
                self.trigger_model = TEASER(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "ecec":
                self.trigger_model = ECEC(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "ecdire":
                self.trigger_model = ECDIRE(self.chronological_classifiers, **self.trigger_params)
            elif self.trigger_model == "edsc":
                self.trigger_model = EDSC(min_length=5, max_length=self.chronological_classifiers.max_length//2, **self.trigger_params)
            elif self.trigger_model == "ects":
                self.trigger_model = ECTS(self.models_input_lengths, **self.trigger_params)
            elif self.trigger_model == "calimera":
                self.trigger_model = CALIMERA(self.cost_matrices, self.models_input_lengths, **self.trigger_params) 
            elif self.trigger_model == "elects":
                self.trigger_model = self.chronological_classifiers # embed trigger model  
            else:
                raise ValueError("Unknown trigger model")
        else:
            self.trigger_model = ProbabilityThreshold(self.cost_matrices, self.models_input_lengths, **self.trigger_params)
            warn("No trigger model given, using base probability"
                 "threshold grid-search by default")

        self._fit_trigger_model(X_trigger, y_trigger)

        if self.trigger_model.alter_classifiers:
            self.new_chronological_classifiers = self.trigger_model.chronological_classifiers # if ECDIRE
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

        chrono_clf = self.chronological_classifiers
        if self.trigger_model.alter_classifiers:
            chrono_clf = self.new_chronological_classifiers

        # Predict
        if self.trigger_model.require_classifiers:
            #classes = self.chronological_classifiers.predict(X)  
            probas = chrono_clf.predict_proba(X)
            classes = probas.argmax(axis=-1)

            if self.trigger_model.require_past_probas:
                past_probas = chrono_clf.predict_past_proba(X)
                triggers, costs = self.trigger_model.predict(X, past_probas)
            else:
                triggers, costs = self.trigger_model.predict(X, probas)

        else:
            classes, triggers, _ = self.trigger_model.predict(X)
            probas, costs = None, None

        return classes, probas, triggers, costs

    def score(self, X, y, return_metrics=False):

        past_trigger = np.zeros((X.shape[0], )).astype(bool)
        trigger_mask = np.zeros((X.shape[0], )).astype(bool)

        all_preds = np.zeros((X.shape[0], )) - 1
        all_t_star = np.zeros((X.shape[0], )) - 1
        all_f_star = np.zeros((X.shape[0], )) - 1

        if self.trigger_model.require_past_probas:
            all_probas = np.array(
                self.chronological_classifiers.predict_past_proba(X)
            )

        for t, l in enumerate(self.models_input_lengths):

            if not self.trigger_model.require_classifiers:
                classes, triggers, all_t_star = self.trigger_model.predict(X)
                def_preds = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]
                all_preds = np.nan_to_num(classes, nan=def_preds)
                all_t_star = np.array(
                    [self.models_input_lengths[np.searchsorted(self.models_input_lengths, val, side='left')] for val in all_t_star]
                )

                all_f_star = np.array([self.cost_matrices[np.where(self.models_input_lengths==t)[0][0]][int(all_preds[i])][y[i]] 
                                       for i, t in enumerate(all_t_star)])
                break
            elif self.trigger_model.require_past_probas: # not call the predict to avoid recomputing all past probas T times
                probas_tmp = all_probas[:, :t+1, :]
                classes = probas_tmp.argmax(axis=-1)[:, -1]
                triggers = self.trigger_model.predict(X[:, :l], probas_tmp)[0]
            else:
                classes, _, triggers, _ = self.predict(X[:, :l])
            
            trigger_mask = triggers
            # already made predictions
            if past_trigger.sum() != 0: 
                trigger_mask[np.where(past_trigger==True)] = False

            all_preds[trigger_mask] = classes[trigger_mask]
            all_t_star[trigger_mask] = l
            all_f_star[trigger_mask] = np.array([self.cost_matrices[t][int(p)][y[trigger_mask][i]]
                                                 for i, p in enumerate(classes[trigger_mask])])

            # update past triggers with current triggers
            past_trigger = past_trigger | triggers

            # if all TS have been triggered 
            if past_trigger.sum() == X.shape[0]:
                break

            if l == self.chronological_classifiers.models_input_lengths[-1]:
                all_t_star[np.where(all_t_star < 0)] = l
                # final prediction is the valid prediction
                all_preds[np.where(all_preds < 0)] = classes[np.where(all_preds < 0)]
                # if no prediction so far, output majority class 
                if np.isnan(all_preds).any():
                    all_preds[np.where(np.isnan(all_preds))] = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]

                all_f_star[np.where(all_f_star < 0)] = np.array([self.cost_matrices[-1][int(p)][y[i]]
                                                                for i, p in enumerate(all_preds[np.where(all_f_star < 0)])])
                break
                
        acc = (all_preds==y).mean()
        earl = np.mean(all_t_star) / X.shape[1]
        avg_score = np.mean(all_f_star)
        #avg_score = average_cost(acc, earl, self.cost_matrices.alpha)

        if return_metrics:
            kappa = cohen_kappa_score(all_preds, y)
            return {
                "accuracy": acc,
                "earliness": earl,
                "average_cost": avg_score,
                "harmonic_mean": hmean((acc, 1-earl)),
                "kappa": kappa, 
                "pred_t_star": all_preds, 
                "t_star": all_t_star,
                "f_star": all_f_star
            }

        return (avg_score, acc, earl) 
    
    def get_post(self, X, y, use_probas=False, return_metrics=False):

        if use_probas and not self.trigger_model.require_classifiers:
            raise ValueError("Unable to estimate probabilities for trigger models"
                             "that doesn't rely on probabilistic classifiers")

        all_f = np.zeros((len(self.models_input_lengths), len(y)))
        all_preds = np.zeros((len(self.models_input_lengths), len(y)))

        for t, l in enumerate(self.models_input_lengths):
            probas = self.chronological_classifiers.predict_proba(X[:, :l])
            classes = np.argmax(probas, axis=-1)
            all_preds[t] = classes
            if use_probas:
                all_f[t] = np.array(
                    [self.cost_matrices[t][:][y[i]] * probas[i] for i in range(len(y))]
                ).sum(axis=-1)
            else:
                if l == self.models_input_lengths[-1]:
                    default_pred = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]
                    all_f[t] = [self.cost_matrices[t][int(p)][y[i]] 
                                if not np.isnan(p) else self.cost_matrices[t][default_pred][y[i]] 
                                for i, p in enumerate(classes)]
                else:
                    all_f[t] = [self.cost_matrices[t][int(p)][y[i]] 
                                if not np.isnan(p) else np.inf
                                for i, p in enumerate(classes)]
        
        t_post_idx = all_f.argmin(axis=0)
        all_t_post = [self.models_input_lengths[idx]
                      for idx in t_post_idx]
        
        all_f_post = [f[t_post_idx[i]] for i, f in enumerate(all_f.T)]
        # if nan, select majority class 
        all_preds_t_post = [int(p[t_post_idx[i]]) if not np.isnan(all_preds[:,i]).all()
                            else np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]
                            for i, p in enumerate(all_preds.T)]

        if return_metrics:
            acc = (all_preds_t_post==y).mean()
            earl = np.mean(all_t_post) / X.shape[1]
            return {
                "accuracy_post": acc,
                "earliness_post": earl,
                "average_cost_post": np.mean(all_f_post),
                "harmonic_mean_post": hmean((acc, 1-earl)),
                "kappa_post": cohen_kappa_score(all_preds_t_post, y),
                "pred_t_post": all_preds_t_post,
                "t_post": all_t_post,
                "f_post": all_f_post
            }

        return all_t_post, all_f_post, all_preds_t_post