import copy
import numpy as np
from scipy.stats import hmean
from warnings import warn

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier

from .classification.chrono_classifier import ClassifiersCollection
#from .deep.deep_classifiers import DeepChronologicalClassifier
from .cost_matrices import CostMatrices

from .trigger._base import BaseTriggerModel
from .trigger import *
from .utils import check_X_y

# from ects_demokritos import ECTS

class EarlyClassifier(BaseEstimator):
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
                 trigger_model=None,
                 cost_matrices=None,
                 trigger_proportion=0.4,
                 prefit_classifiers=False, 
                 random_state=44):
        
        self.cost_matrices = cost_matrices
        self.chronological_classifiers = copy.deepcopy(chronological_classifiers)
        self.prefit_classifiers = prefit_classifiers

        self.trigger_model = trigger_model
        self.trigger_proportion = trigger_proportion

        self.random_state = random_state

    # Properties are used to give direct access to the 
    # chronological classifiers and trigger models arguments.
    @property
    def timestamps(self):
        return self.cost_matrices.timestamps
    
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

    def _fit_classifiers(self, X, y):
        self.chronological_classifiers.fit(X, y, self.cost_matrices)
        return self

    def _fit_trigger_model(self, X, y):
        #X_pred = np.stack([self.chronological_classifiers.predict_proba(X[:, :length])
        #                   for length in self.chronological_classifiers.models_input_lengths], axis=1)
        
        if self.trigger_model.require_classifiers:
            X_probas = np.stack(self.chronological_classifiers.predict_past_proba(X, self.cost_matrices))
            self.trigger_model.fit(X, X_probas, y, self.cost_matrices)
        else:
            self.trigger_model.fit(X, None, y, self.cost_matrices)

    def fit(self, X, y):
        """
        Fit the early classifier to given training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            The input samples.
        y : ndarray
            Target labels relative to X.

        Returns
        -------
        self : `EarlyClassifier`
            Fitted Estimator
        """

        # DEFINE THE SEPARATION INDEX FOR VALIDATION DATA
        if self.trigger_proportion != 0 and not self.prefit_classifiers:
            X_clf, X_trigger, y_clf, y_trigger = train_test_split(
                X, y, test_size=self.trigger_proportion, random_state=self.random_state
            )
        else:
            X_clf = X_trigger = X
            y_clf = y_trigger = y
        
        # FIT CLASSIFIERS
        if self.chronological_classifiers is not None:
            if not isinstance(self.chronological_classifiers, ClassifiersCollection): #and \
                #not isinstance(self.chronological_classifiers, DeepChronologicalClassifier):
                raise ValueError(
                    "Argument 'chronological_classifiers' should be an instance of class 'ChronologicalClassifiers'.")
        else:
            if self.trigger_model.require_classifiers:
                self.chronological_classifiers = ClassifiersCollection(
                    base_classifier=HistGradientBoostingClassifier(),
                    learned_timestamps_ratio=0.05,
                    min_length=1
                )
            #self.chronological_classifiers = DeepChronologicalClassifier()

        if not self.cost_matrices:
            self.cost_matrices = CostMatrices(timestamps=self.timestamps, 
                                              n_classes=len(np.unique(y)), 
                                              alpha=0.5)
            warn("No cost matrices defined, using alpha = 0.5 by default")

        if not self.prefit_classifiers:
            self._fit_classifiers(X_clf, y_clf)

        # FIT TRIGGER MODEL
        if self.trigger_model is not None:
                if not isinstance(self.trigger_model, BaseTriggerModel):
                     raise ValueError(
                          "Argument `trigger_model` shloud inherits from the `BaseTriggerModel` class.")
        else:
            warn("No `trigger_model` defined, setting default trigger to Proba_Threshold.")
            self.trigger_model = ProbabilityThreshold(self.timestamps)

        self._fit_trigger_model(X_trigger, y_trigger)

        if self.trigger_model.alter_classifiers:
            self.new_chronological_classifiers = self.trigger_model.chronological_classifiers # if ECDIRE
        # self.non_myopic = True if issubclass(type(self.trigger_model), NonMyopicTriggerModel) else False

        return self

    def predict_proba(self, X):
        return self.chronological_classifiers.predict_proba(X, self.cost_matrices)

    def predict(self, X):
        """
        Predicts the class, class probabilities vectors, trigger indication and expected costs of the time series
        contained in X.

        Parameters
        ----------
        X: array-like 
            The input time series, potentially of various size.

        Returns
        -------
        classes : ndarray
            Array containing the predicted class of each time series in X.
        probas: np.ndarray
            Array containing the predicted class probabilities vectors of each time series in X.
        triggers: np.ndarray
            Array of booleans indicating whether to trigger the decision immediately with the current prediction
            (True) or to wait for more data (False) for each time series in X.
        """

        chrono_clf = self.chronological_classifiers
        if self.trigger_model.alter_classifiers:
            chrono_clf = self.new_chronological_classifiers

        if self.trigger_model.require_classifiers:
            #classes = self.chronological_classifiers.predict(X)  
            probas = chrono_clf.predict_proba(X, self.cost_matrices)
            classes = probas.argmax(axis=-1)

            if self.trigger_model.require_past_probas:
                past_probas = chrono_clf.predict_past_proba(X, self.cost_matrices)
                triggers = self.trigger_model.predict(X, past_probas, self.cost_matrices)
            else:
                triggers = self.trigger_model.predict(X, probas, self.cost_matrices)

        else:
            classes, triggers, _ = self.trigger_model.predict(X, None, self.cost_matrices)
            probas = None

        return classes, probas, triggers

    def score(self, X, y, return_metrics=False):

        past_trigger = np.zeros((X.shape[0], )).astype(bool)
        trigger_mask = np.zeros((X.shape[0], )).astype(bool)

        all_preds = np.zeros((X.shape[0],))-1
        all_t_star = np.zeros((X.shape[0],))-1
        all_f_star = np.zeros((X.shape[0],))-1

        if self.trigger_model.require_past_probas:
            all_probas = np.array(
                self.chronological_classifiers.predict_past_proba(X)
            )

        for t, l in enumerate(self.timestamps):

            if not self.trigger_model.require_classifiers:
                classes, triggers, all_t_star = self.trigger_model.predict(X, None, self.cost_matrices)
                def_preds = np.unique(y)[np.argmax(self.trigger_model.class_prior)]
                all_preds = np.nan_to_num(classes, nan=def_preds)
                all_t_star = np.array(
                    [self.timestamps[np.searchsorted(self.timestamps, val, side='left')] for val in all_t_star]
                )

                all_f_star = np.array([self.cost_matrices[np.where(self.timestamps==t)[0][0]][int(all_preds[i])][y[i]] 
                                       for i, t in enumerate(all_t_star)])
                break

            elif self.trigger_model.require_past_probas: # not call the predict to avoid recomputing all past probas T times
                probas_tmp = all_probas[:, :t+1, :]
                classes = probas_tmp.argmax(axis=-1)[:, -1]
                triggers = self.trigger_model.predict(X[:, :l], probas_tmp, self.cost_matrices)
            else:
                classes, _, triggers = self.predict(X[:, :l])
            
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

            if l == self.chronological_classifiers.timestamps[-1]:
                all_t_star[np.where(all_t_star < 0)] = l
                # final prediction is the valid prediction
                all_preds[np.where(all_preds < 0)] = classes[np.where(all_preds < 0)]
                # if no prediction so far, output majority class 
                if np.isnan(all_preds).any():
                    all_preds[np.where(np.isnan(all_preds))] = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]

                all_f_star[np.where(all_f_star < 0)] = np.array([self.cost_matrices[-1][int(p)][y[np.where(all_f_star < 0)][i]]
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

        all_f = np.zeros((len(self.timestamps), len(y)))
        all_preds = np.zeros((len(self.timestamps), len(y)))

        for t, l in enumerate(self.timestamps):
            probas = self.chronological_classifiers.predict_proba(X[:, :l], self.cost_matrices)
            classes = np.argmax(probas, axis=-1)
            all_preds[t] = classes
            if use_probas:
                all_f[t] = np.array(
                    [self.cost_matrices[t][:][y[i]] * probas[i] for i in range(len(y))]
                ).sum(axis=-1)
            else:
                if l == self.timestamps[-1]:
                    default_pred = np.unique(y)[np.argmax(self.chronological_classifiers.class_prior)]
                    all_f[t] = [self.cost_matrices[t][int(p)][y[i]] 
                                if not np.isnan(p) else self.cost_matrices[t][default_pred][y[i]] 
                                for i, p in enumerate(classes)]
                else:
                    all_f[t] = [self.cost_matrices[t][int(p)][y[i]] 
                                if not np.isnan(p) else np.inf
                                for i, p in enumerate(classes)]
        
        t_post_idx = all_f.argmin(axis=0)
        all_t_post = [self.timestamps[idx]
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