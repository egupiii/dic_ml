import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Create a class of linear regression from scratch
class ScratchLinearRegression():
    """
    Implementation of linear regression from scratch

    Parameters
    ----------
    num_iter: int
        The number of iteration

    lr: float
        Learning rate

    bias: bool
        True if input the bias term

    verbose: bool
        True if output the learning process


    Attributes
    ----------
    self.coef_: ndarray whose shape is (n_features,)
        parameters

    self.loss: ndarray whose shape is (self.iter,)
        records of loss on train dataset

    self.val_loss: ndarray whose shape is (self.iter,)
        records of loss on validation dataset
    """

    def __init__(self, num_iter, lr, bias, verbose):
        # Record hyperparameters as attribute
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose

        # Prepare arrays for recording loss
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)


    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit linear regression. In a case of inputting validation dataset, return the loss and the accuracy of
        datasets per iteration.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Features of train dataset

        y: ndarray whose shape is (n_samples,)
            Correct values of train dataset

        X_val: ndarray whose shape is (n_samples,n_features)
            Features of validation dataset

        y_val: ndarray whose shape is (n_samples,)
            Correct values of validation dataset
        """

        ###print("fit-1, X=",X.shape)   # (1168,2)
        ###print("fit-2, y=",y.shape)   # (1168,)
        ###print("fit-101, X_val=",X_val.shape)   # (292,2)
        ###print("fit-102, y_val=",y_val.shape)   # (292,)

        # Change the vectors to a matrix
        y = y.reshape(len(y), 1)
        if y_val is not None:
            y_val = y_val.reshape(len(y_val), 1)

        # Add a bias if self.bias is True
        if self.bias == True:
            # Create arrays of biases
            X_bias = np.array([1 for _ in range(X.shape[1])])
            y_bias = np.array([1 for _ in range(y.shape[1])])
            ###print("fit-3, X_bias=",X_bias.shape)   # (1168,)
            ###print("fit-4, y_bias=",y_bias.shape)   # (1168,)
            # Add the biases
            X = np.vstack((X_bias, X))
            y = np.vstack((y_bias, y))

        # Transform dataframes to move their features to rows
        X = X.T
        y = y.T
        if (X_val is not None) and (y_val is not None):
            X_val = X_val.T
            y_val = y_val.T

        ###print("fit-5, X=",X.shape)   # (2,1168)
        ###print("fit-6, y=",y.shape)   # (1,1168)
        ###if (X_val is not None) and (y_val is not None):
        ###print("fit-103, X_val=",X_val.shape)   # (2,292)
        ###print("fit-104, y_val=",y_val.shape)   # (1,292)

        # Set a hypothesis parameter randomly and transform it
        self.coef_ = np.random.randn(X.shape[0])
        self.coef_ = self.coef_.reshape(len(self.coef_), 1)
        ###print("fit-7, self.coef_=",self.coef_.shape)   # (2,1)

        # Update the theta and get loss of train dataset
        for i in range(self.iter):
            # Update the parameter
            self.coef_ = self._gradient_descent(X, y)
            ###print("fit-8, self.coef_=",self.coef_.shape)   # (2,1)
            # Compute the mean square mean
            mse = self._compute_cost(X, y)
            ###print("fit-9, mse=",mse.shape)   # ()
            # Record the errors
            self.loss[i] = mse
            # Return the loss if verbose is True
            if self.verbose:
                print(self.loss[i])

            # Get loss of validation datasets
            if (X_val is not None) and (y_val is not None):
                # Get the mean square error
                val_mse = self._compute_cost(X_val, y_val)
                # Record the errors
                self.val_loss[i] = val_mse
                # Return the loss if verbose is True
                if self.verbose:
                    print(self.val_loss[i])


    def predict(self, X):
        """
        Predict by using linear regression

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Samples


        Returns
        ----------
        ndarray whose shape is (n_samples,1)
            Results of the prediction by using linear regression
        """

        # Add a bias if self.bias is True
        if self.bias == True:
            X_bias = np.array([1 for _ in range(X.shape[1])])
            X = np.vstack((X_bias, X))

        ###print("predict-1, self.coef_=",self.coef_.shape)
        ###print("predict-2, X=",X.shape)

        # Predict train dataset
        y_pred = self._linear_hypothesis(X.T)  # (1,2) * (2,293)

        return y_pred


    # Create a definition of hypothesis function of linear regression
    def _linear_hypothesis(self, X):
        """
        Return hypothesis function of linear regression

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Train dataset

        Returns
        ----------
        ndarray whose shape is (n_samples,1)
            Results of the prediction by hypothesis function of linear regression
        """

        # Compute the hypothesis function
        y_pred = np.dot(self.coef_.T, X)   # (1,2) * (2,1168)
        ###print("_linear_hypothesis-1, y_pred=",y_pred.shape)   # (1,1168)

        return y_pred


    # Create a definition to compute the mean square error
    def _compute_cost(self, X, y):
        """
        Compute the mean square error. Import the "MSE" definition.

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            train dataset

        y: ndarray whose shape is (n_samples,1)
            correct value


        Returns
        ----------
        ndarray whose shape is (1,)
            mean square error
        """

        y_pred = self._linear_hypothesis(X)

        return self.MSE(y_pred, y)


    # Create a definition of the mean square error
    def MSE(self, y_pred, y):
        """
        Return the mean square error

        Parameters
        ----------
        y_pred: ndarray whose shape is (n_samples,)
            predited value

        y: ndarray whose shape is (n_samples,)
            correct value


        Returns
        ----------
        mse: numpy.float
            mean square error
        """

        # Compute an error
        error = y_pred - y
        ###print("MSE-1, error=",error.shape)   # (1,1168)

        # Sum errors
        sum_errors = np.sum(error ** 2)

        # Return the mean square error devided by 2
        return sum_errors / (2 * y.shape[1])


    # Create a definition to fit datasets by steepest descent method
    def _gradient_descent(self, X, y):
        """
        Fit datasets by steepest descent method

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            train dataset

        y: ndarray whose shape is (n_samples,1)
            correct value


        Returns
        ----------
        ndarray whose shape is (1,)
            parameter(weight)
        """

        # Predict train dataset
        y_pred = self._linear_hypothesis(X)  # (1,2) * (2,1168)
        ###print("_gradient_decsent-1, y_pred=",y_pred.shape)  # (1,1168)

        ###print("_gradient_decsent-1, y=",y.shape)   # (1,1168)

        # Compute the error and the mean square error
        error = y_pred - y  # (1,1168)
        ###print("_gradient_decsent-2, error=",error.shape)   # (1,1168)

        # Compute the gradient
        grad = np.dot(X, error.T)  # (2,1168) * (1168,1)
        ###print("_gradient_decsent-3, grad=",grad.shape)   # (2,1)

        # Update the parameter
        return self.coef_ - self.lr * grad / y.shape[1]


    # Plot learning records
    def plot_learning_record(self):
        plt.plot(self.loss, label="loss")
        plt.plot(self.val_loss, label="val_loss")

        plt.title("Learning Records")
        plt.xlabel("Number of Iterrations")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.legend()
        plt.show()