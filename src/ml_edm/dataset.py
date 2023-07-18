import tsfel
import numpy as np
from pandas import DataFrame
from warnings import warn


def get_time_series_lengths(X):
    """
    Parameters:
        X: numpy.ndarray
            An array of size (N*max_T) where N is the number of time series, max_T the max number of measurements in a
            time series and where empty values are filled with nan. Can also be a pandas DataFrame or a list of lists.
    Returns:
        A numpy array containing the index of the last numerical value+1 for each row in X, corresponding to the length
        of each time series.
    """
    not_nan_coordinates = np.nonzero(np.logical_not(np.isnan(X)))
    not_nan_count_per_row = np.unique(not_nan_coordinates[0], return_counts=True)[1]
    not_nan_indices_per_row = np.split(not_nan_coordinates[1], np.cumsum(not_nan_count_per_row)[:-1])
    return np.array([np.max(array) + 1 for array in not_nan_indices_per_row])


def extract_features(X):  # TODO: Replace with a better feature extraction.
    """
    Uses the tsfel library to perform feature extraction on a dataset of time series.
    Feature extraction is quite slow, wordy, does not work with time series with less than 12 measurements and does not
    seem very effective. We do not recommend its use without experimentation yet.

    Parameters:
        X: ndarray
            Dataset of time series of various sizes to extract features from. An array of size (N*max_T) where N is the
            number of time series, max_T the max number of measurements in a time series and where empty values are filled
            with nan. Can also be a pandas DataFrame or a list of lists.
    Returns:
        X_concat: ndarray
         Array of size (N, max_T+max_F)  where N is the number of time series, max_T the max number of measurements in a
          time series and max_F is the max number of extracted features. This dataset is obtained by concatenating the
          extracted features to the original time series for each line. Empty values are filled with nan.
    """
    if isinstance(X, list):
        max_series_length = np.max([len(time_series) for time_series in X])
        padding = np.full((len(X), max_series_length), np.nan)
        for i, time_series in enumerate(X):
            padding[i, :len(time_series)] = time_series
        X = padding
    elif isinstance(X, DataFrame):
        X = X.to_numpy
    elif not isinstance(X, np.ndarray):
        raise TypeError("X should be a list, array, or DataFrame of time series.")

    # Get length of each time series
    time_series_lengths = get_time_series_lengths(X)

    # Truncate array for feature extraction
    X_truncated = np.array([time_series[:time_series_lengths[i]] for i, time_series in enumerate(X)])

    # Selecting time series on which to extract features since the extraction cannot be done on time series of size
    # less than 12
    too_short_time_series = np.nonzero(time_series_lengths < 12)[0]
    X_extracted = X_truncated[time_series_lengths >= 12]

    # Extracting features
    cfg_file = tsfel.get_features_by_domain()
    if len(X_extracted) != 0:
        X_extracted = tsfel.time_series_features_extractor(cfg_file, X_extracted).to_numpy()
    # Combining extracted features to the original time series and reinserting the time series that could not be used
    # for feature extraction.
    X_concat = []
    skipped = 0
    for i, time_series in enumerate(X_truncated):
        if np.isin(i, too_short_time_series):
            X_concat.append(time_series)
            skipped += 1
        else:
            # We remove nan features on predict to remove errors caused by using feature extraction on time series of
            # inconsistent size.
            cleaned_features = list(filter(lambda f: str(f) != 'nan', X_extracted[i-skipped]))
            X_concat.append(np.concatenate((time_series, cleaned_features), 0))

    # Converting back to np arrays
    max_input_length = np.max([len(input_) for input_ in X_concat])
    padding = np.full((len(X), max_input_length), np.nan)
    for i, input_ in enumerate(X_concat):
        padding[i, :len(input_)] = input_

    X_concat = padding

    # Warning the user if some feature extraction could not be done.
    if len(too_short_time_series) > 0:
        warn("Could not extract features from time series with length inferior to 12.")
    return X_concat
