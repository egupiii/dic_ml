import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



# Create a class of logistic regression from scratch
class ScratchLogisticRegression():
    """
    Implement logistic regression from scratch.

    Parameters
    ----------
    num_iter: int
        The number of iteration

    lr: float
        Learning rate

    reg: float
        Regularization parameter

    bias: bool
        True if input the bias term

    verbose: bool
        True if output the learning process


    Attributes
    ----------
    self.coef_: ndarray, shape(n_features,)
        parameters

    self.loss: ndarray, shape(self.iter,)
        records of loss on train dataset

    self.val_loss: ndarray, shape(self.iter,)
        records of loss on validation dataset
    """

    def __init__(self, num_iter, lr, reg, bias, verbose):
        # Record hyperparameters as attribute
        self.iter = num_iter
        self.lr = lr
        self.reg = reg
        self.bias = bias
        self.verbose = verbose

        # Prepare arrays for recording loss
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)


    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit datasets by logistic regression. In a case of inputting validation dataset, return the loss
        and the accuracy of datasets per iteration.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Features of train dataset

        y: ndarray, shape(n_samples,)
            Correct values of train dataset

        X_val: ndarray, shape(n_samples,n_features)
            Features of validation dataset

        y_val: ndarray, shape(n_samples,)
            Correct values of validation dataset
        """

        # Change objective vectors to matrixes
        y = y.reshape(len(y), 1)
        if (X_val is not None) and (y_val is not None):
            y_val = y_val.reshape(len(y_val), 1)

        # Add a bias term if self.bias is True
        if self.bias == True:
            # Create arrays of biases
            X_bias = np.array([1 for _ in range(X.shape[0])])
            if (X_val is not None) and (y_val is not None):
                X_val_bias = np.array([1 for _ in range(X_val.shape[0])])

            # Change the vectors to matrixes
            X_bias = X_bias.reshape(len(X_bias), 1)
            if (X_val is not None) and (y_val is not None):
                X_val_bias = X_val_bias.reshape(len(X_val_bias), 1)

            # Add the biases
            X = np.hstack((X_bias, X))
            if (X_val is not None) and (y_val is not None):
                X_val = np.hstack((X_val_bias, X_val))

        # Change the arrays to lists
        y = y.tolist()
        if (X_val is not None) and (y_val is not None):
            y_val = y_val.tolist()

        # Change the original values of the lists to only 2 kinds of values, 0 and 1
        y = [1 if i == max(y) else 0 for i in y]
        if (X_val is not None) and (y_val is not None):
            y_val = [1 if i == max(y_val) else 0 for i in y_val]

        # Change the lists to arrays
        y = np.array(y)
        if (X_val is not None) and (y_val is not None):
            y_val = np.array(y_val)

        # Change the vectors to matrixes
        y = y.reshape(1, len(y))
        if (X_val is not None) and (y_val is not None):
            y_val = y_val.reshape(1, len(y_val))

        # Transform dataframes to move their features to rows
        X = X.T
        if (X_val is not None) and (y_val is not None):
            X_val = X_val.T

        # Set a hypothesis parameter randomly and transform it
        self.coef_ = np.random.randn(X.shape[0])

        # Change the vector to a matrix
        self.coef_ = self.coef_.reshape(len(self.coef_), 1)

        # Update the parameter and get loss of train dataset
        for i in range(self.iter):
            # Update the parameter
            self.coef_ = self._gradient_descent(X, y)

            # Compute the cross entropy
            cross_entropy = self._compute_cost(X, y)

            # Record the errors
            self.loss[i] = cross_entropy

            # Return the loss if verbose is True
            if self.verbose:
                print(self.loss[i])

            # Get loss of validation datasets
            if (X_val is not None) and (y_val is not None):
                # Get the cross entropy
                val_cross_entropy = self._compute_cost(X_val, y_val)

                # Record the errors
                self.val_loss[i] = val_cross_entropy

                # Return the loss if verbose is True
                if self.verbose:
                    print(self.val_loss[i])


    def predict(self, X):
        """
        Predict by logistic regression.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Samples


        Returns
        ----------
        ndarray, shape(n_samples,1)
            Results of the prediction
        """

        y_pred = self.predict_proba(X)

        # Change a probability that is more than 0.5 to 1 and a probability that is less than or equals to 0.5 to 0
        pred_list = []
        for i in y_pred[0]:
            pred_list.append(round(i))

        # Change the values of the list to integers
        pred_list = list(map(int, pred_list))

        return pred_list


    def predict_proba(self, X):
        """
        Probability estimation for logistic regression.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Samples


        Returns
        ----------
        float
            Probability of
        """

        # Add a bias if self.bias is True
        if self.bias == True:
            # Create arrays of biases
            X_bias = np.array([1 for _ in range(X.shape[0])])

            # Change the vectors to a matrix
            X_bias = X_bias.reshape(len(X_bias), 1)

            # Add the biases
            X = np.hstack((X_bias, X))

        # Predict train dataset
        y_pred = self._linear_hypothesis(X.T)

        return y_pred


    # Create a definition of sigmoid function
    def _sigmoid_function(self, z):
        """
        Return sigmoid function.

        Parameters
        ----------
        z: int
            Index of natural logarithm of sigmoid function

        Returns
        ----------
        float
            Results of computation by sigmoid function
        """

        # Compute sigmoid function
        return 1 / (1 + math.e ** (-z))


    # Create a definition of hypothesis function
    def _linear_hypothesis(self, X):
        """
        Return hypothesis function

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Train dataset

        Returns
        ----------
        ndarray, shape(n_samples,1)
            Results of the prediction by hypothesis function
        """

        # Compute an index of linear hypothesis
        z = np.dot(self.coef_.T, X)

        # Compute the hypothesis function
        y_pred = self._sigmoid_function(z)

        return y_pred


    # Create a definition to compute the cross entropy
    def _compute_cost(self, X, y):
        """
        Compute the cross entropy. Import the "cross_entropy" definition.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            train dataset

        y: ndarray, shape(n_samples,1)
            correct value


        Returns
        ----------
        ndarray, shape(1,)
            cross entropy
        """

        y_pred = self._linear_hypothesis(X)

        return self.cross_entropy(y_pred, y)


    # Create a definition of the cross entropy
    def cross_entropy(self, y_pred, y):
        """
        Compute the cross entropy.

        Parameters
        ----------
        y_pred: ndarray, shape(n_samples,)
            predited value

        y: ndarray, shape(n_samples,)
            correct value


        Returns
        ----------
        cross_entropy: numpy.float
            cross entropy
        """

        # Compute a probability that equals to 1
        prob1 = -y * np.log(y_pred)

        # Compute a probability that equals to 0
        prob0 = (1 - y) * np.log(1 - y_pred)

        # Compute the joint probability
        joint_prob = prob1 - prob0

        # Sum the joint probabilities
        sum_joint_probs = np.sum(joint_prob)

        # Compute a regularization term
        term_reg = self.reg / (2 * y.shape[1]) * np.sum(self.coef_ ** 2)

        # Compute the cross entropy
        return sum_joint_probs / y.shape[1] + term_reg


    # Create a definition to fit datasets by steepest descent method
    def _gradient_descent(self, X, y):
        """
        Fit datasets by steepest descent method

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            train dataset

        y: ndarray, shape(n_samples,1)
            correct value


        Returns
        ----------
        ndarray, shape(1,)
            parameter(weight)
        """

        # Predict train dataset
        y_pred = self._linear_hypothesis(X)

        # Compute the error
        error = y_pred - y

        # Compute the gradient
        grad = np.dot(X, error.T)

        # Sum gradients
        sum_grads = np.sum(grad)

        # Compute the regularization term
        reg_term = self.reg / y.shape[1] * self.coef_

        # Update the parameter
        if self.bias == False:
            return self.coef_ - self.lr * (grad / y.shape[1] + reg_term)
        else:
            # Change the (1,1) element to 0
            reg_term[0, 0] = 0
            return self.coef_ - self.lr * (grad / y.shape[1] + reg_term)


    # Plot learning records
    def plot_learning_record(self):
        """
        Plot learning records.
        """

        plt.plot(self.loss, label="loss")
        plt.plot(self.val_loss, label="val_loss")

        plt.title("Learning Records")
        plt.xlabel("Number of Iterrations")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.legend()
        plt.show()


    # Compute index values
    def compute_index_values(self, X, y):
        """
        Compute Index values.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Features of train dataset

        y: ndarray, shape(n_samples,)
            Correct values of train dataset
        """

        y_pred = self.predict(X)

        # Change values of y to only 2 kinds of values, 0 and 1
        y = [1 if i == max(y) else 0 for i in y]

        # Return index values
        print("accuracy score: ", accuracy_score(y, y_pred))
        print("precision score: ", precision_score(y, y_pred))
        print("recall score: ", recall_score(y, y_pred))
        print("f1 score: ", precision_score(y, y_pred))
        print("confusion matrix:")
        print(confusion_matrix(y, y_pred))


    def decision_boundary(self, X, y, step=0.01, title="Decision Boundary", xlabel="Number of Iteration", ylabel="Loss",
                          target_names=["setosa", "virginica"]):
        """
        Plot a decision boundary of a model fitting binary classification by 2-dimentional features.

        Parameters
        ----------------
        X : ndarray, shape(n_samples, 2)
            Features of train dataset

        y : ndarray, shape(n_samples,)
            Correct values of train dataset

        step : float, (default : 0.1)
            Set intervals to compute the prediction

        title : str
            Input the title of the graph

        xlabel, ylabel : str
            Input names of each axis

        target_names= : list of str
            Input a list of the legends
        """

        # Setting
        scatter_color = ["r", "b"]
        contourf_color = ["pink", "skyblue"]
        n_class = 2

        # Predict
        mesh_f0, mesh_f1 = np.meshgrid(np.arange(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5, step),
                                       np.arange(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5, step))
        mesh = np.c_[np.ravel(mesh_f0), np.ravel(mesh_f1)]
        pred = self.predict_proba(mesh).reshape(mesh_f0.shape)

        # Plot
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.contourf(mesh_f0, mesh_f1, pred, n_class - 1, cmap=ListedColormap(contourf_color))
        plt.contour(mesh_f0, mesh_f1, pred, n_class - 1, colors='y', linewidths=3, alpha=0.5)
        for i, target in enumerate(set(y)):
            plt.scatter(X[y == target][:, 0], X[y == target][:, 1], s=80, color=scatter_color[i], label=target_names[i],
                        marker='o')
        patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]
        plt.legend(handles=patches)
        plt.legend()
        plt.show()