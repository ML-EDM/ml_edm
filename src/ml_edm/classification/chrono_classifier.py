import os
import copy
import numpy as np

from ._base import BaseTimeClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from .features_engineering.features_extraction import Feature_extractor
#from ..trigger import *
#from ..utils import *

from warnings import warn

# TODO : Multivariate data
# TODO : Enrich / change feature extraction
# TODO : Add calibration
# TODO : deal with NaN in dataset ? (Although HGBoost is nan compatible)
# TODO : Add set_params, setters etc... Check integrity issues doing so
# TODO : Optimize decision threshold
# TODO : Verbose, loadbar?
# TODO : implement sparse matrix compatibility
# TODO : Make classes sklearn estimators

class ClassifiersCollection(BaseTimeClassifier):
    """
    A class containing a list of classifiers to train on time series of incrementing lengths. Can be used along with a
    trigger model object such as an EconomyGamma instance to do tasks of early classification.

    Parameters:
    -----------

        base_classifier : classifier instance, default=sklearn.ensemble.HistGradientBoostingClassifier()
            Classifier instance to be cloned and trained for each input length.
        timestamps : numpy.ndarray, default=None
            Array containing the numbers of time measurements/input length that each classifier is trained on.
            Argument 'nb_classifiers' is deduced from the length of this list.
        sampling_ratio : float, default = None
            Ignored if 'timestamps' is defined.
            Foat number between 0 and 1, define frequency at which 'timestamps' are spaced.
        min_length : int, default = None
            Define the minimum serie length for the first classifier to operate on.
        feature_extraction : dict, default = None 
            Either a dictionnary containg one of method ['minirocket', 'weasel2.0', 'tsfresh'] and eventually a 'params'
            to define method parametes as a dict and a 'path' key to define where to save features matrices if desired.
        calibration : boolean, default = True
            Whether or not to use post-hoc calibration (Platt scaling) for each classifier.
        classifiers : numpy.ndarray, default=None
            List or array containing the classifier instances to be trained. Argument 'nb_classifiers' is deduced from
            the length of this list.

    Attributes:
    -----------

        nb_classifiers: int, default=20
            Number of classifiers to be trained. If the number is inferior to the number of measures in the training
            time series, the models input lengths will be equally spaced from max_length/n_classifiers to max_length.
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
                 timestamps=None,
                 sampling_ratio=None,
                 min_length=None,
                 feature_extraction=None,
                 calibration=True,
                 classifiers=None,
                 random_state=44):  
        
        super().__init__(timestamps, 
                         sampling_ratio,
                         min_length)
        
        self.base_classifier = base_classifier
        self.classifiers = classifiers

        self.feature_extraction = feature_extraction
        self.calibration = calibration
        self.random_state = random_state

    def __getitem__(self, item):
        return self.classifiers[item]

    def __len__(self):
        return self.nb_classifiers

    def _fit(self, X, y, *args, **kwargs):
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
        # check classifiers list validity
        if self.classifiers is not None:
            if not isinstance(self.classifiers, list):
                raise TypeError("Argument 'classifiers' should be a list of classifier objects.")
            if len(self.classifiers) == 0:
                raise ValueError("List argument 'classifiers' is empty.")
            if self.base_classifier is not None:
                warn("Both base_classifier and classifiers arguments are defined,"
                     " in that case the base_classifier argument will be ignored.")
        else:
            if self.base_classifier is None:
                self.base_classifier = HistGradientBoostingClassifier(random_state=self.random_state)
                warn("Using 'base_classifier = HistGradientBoostingClassifier() by default.")

            self.classifiers = [copy.deepcopy(self.base_classifier) for _ in range(self.nb_classifiers)]

        # feature_extraction check
        if self.feature_extraction:
            if isinstance(self.feature_extraction, dict) and \
                'method' in self.feature_extraction.keys():
                
                if self.feature_extraction['method'] not in ['minirocket', 'weasel2.0', 'tsfresh']:
                    raise ValueError("Argument 'method' from 'feature_extraction' should be one of "
                                    "['minirocket', 'weasel2.0', 'tsfresh']")
            elif not isinstance(self.feature_extraction, str):
                raise ValueError("Argument 'feature_extraction' should be one of dictionnary "
                                "or string (path from which to retreive already computed features)")

        # FEATURE EXTRACTION AND FITTING
        self.extractors = []
        for i, ts_length in enumerate(self.timestamps):
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
                Xt_clf, y_clf = (Xt.reshape(len(X),-1), y) # keep all training samples
                #Xt_clf, _, y_clf, _ = train_test_split(Xt.reshape(len(X),-1), y, test_size=0.3, stratify=y,
                #                                       random_state=self.random_state) # throw calib samples

            self.classifiers[i].fit(Xt_clf, y_clf, **kwargs)
            
            if self.calibration:
                calib_clf = CalibratedClassifierCV(self.classifiers[i], cv='prefit')
                self.classifiers[i] = calib_clf.fit(X_calib, y_calib)

        return self

    def _predict_proba(self, grouped_X, cost_matrices=None):
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
        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.timestamps[0]:
                predictions.append(np.ones((len(series), len(self.class_prior))) * self.class_prior)
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.timestamps == length
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
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return np.vstack(predictions)
    
    def _predict_past_proba(self, grouped_X, cost_matrices=None):

        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.timestamps[0]:
                priors = np.ones((len(series), len(self.class_prior))) * self.class_prior
                predictions.append(priors)
                returned_priors = True
            else:
                clf_idx = np.where(
                    self.timestamps == length
                )[0][0]
                if length != self.timestamps[0]:
                    series = np.array(series) # allow for slicing
                    
                    if self.feature_extraction and os.path.isdir(str(self.feature_extraction)):
                        partial_series = [np.load(self.feature_extraction+f"/features_{j}.npy") 
                                          for j in range(clf_idx+1)]
                    elif self.feature_extraction:
                        partial_series = [
                            self.extractors[j].transform(
                            series[:, :self.timestamps[j]])
                            for j in range(clf_idx+1)
                        ]
                    else:
                        partial_series = [series[:, :self.timestamps[j]].reshape(len(series),-1)
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
        if returned_priors:
            warn("Some time series are of insufficient length for prediction ;"
                 " returning prior probabilities instead.")

        return predictions

