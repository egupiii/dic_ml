import numpy as np


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

    # Compute the number of rows that we'll extract
    X_rows = int(np.round(len(X) * train_size))
    y_rows = int(np.round(len(y) * train_size))

    # Shuffle the number
    X_array = np.random.choice(np.arange(len(X)), X_rows, replace=False)
    y_array = np.random.choice(np.arange(len(y)), y_rows, replace=False)

    # Change the arrays to lists
    X_list = X_array.tolist()
    y_list = y_array.tolist()

    # Split X and y
    X_train_list = [X[_] for _ in X_list]
    X_test = np.delete(X, [X[_] for _ in X_list])

    y_train_list = [y[_] for _ in y_list]
    y_test = np.delete(y, [y[_] for _ in y_list])

    # Change the train lists to arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    return X_train, X_test, y_train, y_test