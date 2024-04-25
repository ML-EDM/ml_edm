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
                #Xt_clf, y_clf = (Xt.reshape(len(X),-1), y) # keep all training samples
                Xt_clf, _, y_clf, _ = train_test_split(Xt.reshape(len(X),-1), y, test_size=0.3, stratify=y,
                                                       random_state=self.random_state) # throw calib samples

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

