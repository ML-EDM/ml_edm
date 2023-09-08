import os

import numpy as np
import pandas as pd


def check_X_y(X, y, 
              equal_length=True):

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

    return X, y

def check_X_probas(X_probas, all_timestamps=False):

    # X_probas
    if isinstance(X_probas, list):
        X_probas = np.array(X_probas)
    elif isinstance(X_probas, pd.DataFrame):
        X_probas = X_probas.to_numpy()
    elif not isinstance(X_probas, np.ndarray):
        raise TypeError(
            "X_probas should be a 2-dimensional list, array or DataFrame of size (N, P) with N the number "
            "of examples and P the number of classes probabilities.")
    
    if X_probas.ndim != 2:
        raise ValueError(
            "X_pred should be a 2-dimensional list, array or DataFrame of size (N, P) with N the number "
            "of examples and P the number of classes probabilities.")
    
    if len(X_probas) == 0:
        raise ValueError("Dataset 'X_probas' to predict triggering on is empty.")

    return X_probas