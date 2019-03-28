import numpy as np
from sklearn.utils import shuffle


def train_test_split(X, y, train_size=0.8):
    """
    Split the train dataset.
    
    Parameters
    ----------
    X: ndarray whose shape is (n_samples,n_features)
      train dataset
    y: ndarray whose shape is (n_samples,)
      correct values
    train_size: float (0 < train_size < 1)
      designate what percentage is the train dataset
    
    Returns
    ----------
    X_train: ndarray whose shape is (n_samples, n_features)
      train dataset
    X_test: ndarray whose shape is (n_samples, n_features)
      validation dataset
    y_train: ndarray whose shape is (n_samples,)
      correct values of the train dataset
    y_test: ndarray whose shape is (n_samples,)
      correct values of validation dataset
    """
    
    # Shuffle
    X, y = shuffle(X, y, random_state=0)
    
    # Change the array to a list
    X = X.tolist()
    
    # Compute the number of rows that we'll extract
    n_rows = int(np.round(len(X) * train_size))
    
    # Split X and y
    X_train = X[:n_rows]
    X_test = X[n_rows:]
    y_train = y[:n_rows]
    y_test = y[n_rows:]
    
    # Change the train lists to arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    return X_train, X_test, y_train, y_test