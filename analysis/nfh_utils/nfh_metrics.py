import numpy as np

def mae(predicted, actual) -> float:
    """Calculate Mean Absolute Error."""
    arr_predicted = np.asarray(predicted)
    arr_actual = np.asarray(actual)
    return np.mean(abs(arr_predicted - arr_actual))

def rmse(predicted, actual) -> float:
    """Calculate Root Mean Squared Error."""
    arr_predicted = np.asarray(predicted)
    arr_actual = np.asarray(actual)
    return np.sqrt(((arr_predicted - arr_actual)**2).mean())

def rmae(predicted, actual, window_size):
    """Calculate Rolling Mean Absolute Error."""
    arr_predicted = np.asarray(predicted)
    arr_actual = np.asarray(actual)

    # Calculate rolling window averages
    rolling_predicted = pd.Series(arr_predicted).rolling(window=window_size, min_periods=window_size).mean()
    rolling_actual = pd.Series(arr_actual).rolling(window=window_size, min_periods=window_size).mean()

    # Calculate the mean absolute error between the rolling averages
    rmae = np.mean(np.abs(rolling_predicted - rolling_actual))
    return rmae
    
