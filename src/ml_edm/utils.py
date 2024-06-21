import numpy as np
import pandas as pd

from sklearn.utils.multiclass import check_classification_targets
from warnings import warn 

def check_X_y(X, y, equal_length=True):

    # Check X and y
    if isinstance(X, list):
        if not equal_length:
            # padd series with different lengths 
            max_length = max([len(x) for x in X])
            padding = np.full((len(X), max_length), np.nan)
            for i, time_series in enumerate(X):
                padding[i, :len(time_series)] = time_series
            X = padding
        else:
            X = np.array(X)
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif not isinstance(X, np.ndarray):
        raise TypeError("X should be a 2-dimensional list, array or DataFrame of size (N, T) "
                        "with N the number of examples and T the number of timestamps.")
    
    if equal_length:
        for x in X:
            if len(x) != len(X[0]):
                raise ValueError("All time series in the dataset should have the same length.")
        
    if len(X) == 0:
        raise ValueError("Dataset to fit 'X' is empty.")
    if X.ndim != 2:
        raise ValueError("X should be a 2-dimensional list, array or DataFrame of size (N, T) with N the number "
                        "of examples and T the number of timestamps.")
    
    if y is not None:
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be a list of labels of size N with N the number of examples in X")
        
        if len(y) != len(X):
            raise ValueError("y should be a list of classes of size N with N the number of examples in X")
        
        check_classification_targets(y)

    return X, y

def check_X_probas(X_probas):

    # X_probas
    if isinstance(X_probas, list):
        X_probas = np.array(X_probas)
    elif isinstance(X_probas, pd.DataFrame):
        X_probas = X_probas.to_numpy()
    elif not isinstance(X_probas, np.ndarray):
        raise TypeError(
            "X_probas should be a 2-dimensional list, array or DataFrame of size (N, K) with N the number "
            "of examples and K the number of classes probabilities.")
    
    if X_probas.ndim != 2:
        raise ValueError(
            "X_probas should be a 2-dimensional list, array or DataFrame of size (N, K) with N the number "
            "of examples and K the number of classes probabilities.")
    
    if len(X_probas) == 0:
        raise ValueError("Dataset 'X_probas' to predict triggering on is empty.")

    return X_probas

def check_X_past_probas(X_past_probas):

    # X_probas
    if isinstance(X_past_probas, list):
        X_past_probas = np.array(X_past_probas)
    elif isinstance(X_past_probas, pd.DataFrame):
        X_past_probas = X_past_probas.to_numpy()
    elif not isinstance(X_past_probas, np.ndarray):
        raise TypeError(
            "X_past_probas should be a 3-dimensional list, array or DataFrame of size (T, N, K) with T the number of timepoints,"
            "N the number of examples and K the number of classes probabilities.")
    
    if X_past_probas.ndim != 3:
        raise ValueError(
            "X_past_probas should be a 3-dimensional list, array or DataFrame of size (T, N, K) with T the number of timepoints,"
            " N the number of examples and K the number of classes probabilities.")
    
    if X_past_probas.shape[1] == 0:
        raise ValueError("Dataset 'X_past_probas' to predict triggering on is empty.")

    return X_past_probas

def check_timestamps(timestamps):

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)
    elif not isinstance(timestamps, np.ndarray):
        raise TypeError("Argument 'timestamps' should be a list or array of positive int.")
    if len(timestamps) == 0:
            raise ValueError("List argument 'timestamps' is empty.")
    for t in timestamps:
        if not (isinstance(t, np.int32) or isinstance(t, np.int64)):
            raise TypeError("Argument 'timestamps' should be a list or array of positive int.")
        if t < 0:
            raise ValueError("Argument 'timestamps' should be a list or array of positive int.")
                
    if len(np.unique(timestamps)) != len(timestamps):
        timestamps = np.unique(timestamps)
        warn("Removed duplicates in argument 'timestamps'.")
    
    if 0 in timestamps:
        timestamps = np.nonzero(timestamps)[0]
        warn("Removed 0 from 'timestamps', first valid timestamps is usually 1.")

    return timestamps