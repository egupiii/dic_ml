import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# Set command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description='linear regression pipeline')

parser.add_argument('--dataset')
parser.add_argument('--model',default="lr",type=str)



# Create a definition of a regression pipeline
def implement_regression_pipeline(args):
    # Get datasets
    df = pd.read_csv(args.dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values


    # Split train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


    if args.model == "lr":
        # Initialize a class
        slr = ScratchLinearRegression(num_iter=200, lr=0.1, bias=True, verbose=True)

        # Fit the model according to the given training data
        slr.fit(X_train, y_train)

        # Predict class labels for samples in X_test
        y_pred = slr.predict(X_test)

        # Returns the mean accuracy on the given test data and labels
        mse = slr.MSE(y_pred, y_test)

    else:
        print("Your model, {} did not work on this definition.".format(args.model))


    # Return the results
    print("y_pred is {}".format(y_pred))
    print("MSE is {}".format(mse))



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

    no_bias: bool
        True if not input the bias term

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
        Fit linear regression. In a case of inputting validation dataset, return loss and accuracy of
        the data per iteration.

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

        # Set hypothesis function
        y = self._linear_hypothesis(X)

        # Get lists of logs of MSE and theta
        for i in range(self.iter):
            # Get the mean square error and the updated theta
            mse = self._gradient_descent(X, y)
            # Record the errors
            self.loss[i] = mse
            # Return the log of losses if let verbose True
            if self.verbose:
                print(self.loss)


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

        # Predict a train dataset
        y_pred = np.dot(X, self.coef_)

        return y_pred


    # Create a definition of hypothesis function of lunear regression
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

        # Set a first theta
        self.coef_ = np.random.randint(X.shape[1])

        # Compute the hypothesis function
        y = np.dot(X, self.coef_)

        return y


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

        # Sum errors
        sum_errors = (error ** 2).sum()

        # Compute the mean square error devided by 2
        mse = sum_errors / (2 * len(y))

        return mse


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

        # Predict a train dataset
        y_pred = np.dot(X, self.coef_)

        # Compute the mean square error
        mse = self.MSE(y_pred, y)

        # Compute the error
        error = y_pred - y

        # Compute the gradient
        grad = np.dot(error.T, X)

        # Update the theta
        self.coef_ -= self.lr * grad / len(y)

        return mse



if __name__ == "__main__":
    # Be run here at first when running the py file

    # Import command-line arguments
    args = parser.parse_args()
    implement_regression_pipeline(args)