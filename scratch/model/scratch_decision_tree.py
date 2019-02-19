import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



# Create a class of decision tree from scratch
class ScratchDecisionTreeClassifier():
    """
    Implement decision tree from scratch.

    Attributes
    ----------
    self.threshold: float
        thresholds

    self.left_label: int
        label of a left tree

    self.right_label: int
        label of a right tree
    """

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit datasets by decision tree.

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

        # Transform arrays to move their features to rows
        X = X.T
        if (X_val is not None) and (y_val is not None):
            X_val = X_val.T

        # Decide a threshold
        self.threshold = self._best_threshold(X, y)


    def predict(self, X):
        """
        Prediction

        Parameters
        ----------
        X: ndarray whose shape is (n_samples,n_features)
            Samples


        Returns
        ----------
        ndarray whose shape is (n_samples,1)
            Results of the prediction
        """

        # Transform arrays to move their features to rows
        X = X.T

        # Label
        y_pred = []
        for x in X[1]:
            if x >= self.threshold:
                y_pred.append(self.left_label)
            else:
                y_pred.append(self.right_label)

        # Return the prediction
        return y_pred


    # Create a definition to compute a Gini impurity
    def _gini_impurity(self, m, n):
        """
        Compute Gini impurities.

        Parameters
        ----------
        m: int
            The number of samples

        n: int
            The number of samples whose class is different with the class that the number of samples is m

        Returns
        ----------
        float
            Gini impurity
        """

        # Compute a Gini impurity
        return 1 - ((m / (m + n)) ** 2 + (n / (m + n)) ** 2)


    def _information_gain(self, m, n, s, t):
        """
        Compute Gini impurities.

        Parameters
        ----------
        m: int
            The number of samples on left side

        n: int
            The number of samples on left side whose class is different with the class that the number of samples is m

        s: int
            The number of samples on right side

        t: int
            The number of samples on right side whose class is different with the class that the number of samples is s

        Returns
        ----------
        float
            Information gain
        """

        # Compute Gini impurities
        left_gini_impurity = self._gini_impurity(m, n)
        right_gini_impurity = self._gini_impurity(s, t)

        # Returns an information gain
        return self._gini_impurity(m + n, s + t) - (m + n) / (m + n + s + t) * left_gini_impurity - (s + t) / (
                    m + n + s + t) * right_gini_impurity


    # Choose the best threshold
    def _best_threshold(self, X, y):
        # Set a temporary information gain
        info_gain = 0

        # Change y to a list
        y = y.tolist()

        # Loop of features
        for f in range(X.shape[0]):
            # Delete duplicate values
            non_duplicates_array = np.unique(X[f, :])

            # Delete the minimum value of the arrays not containing duplicate values
            non_dupli = np.delete(non_duplicates_array, non_duplicates_array.min())

            # Loop of samples
            for s in range(len(non_dupli)):
                # Update the threshold
                temporary_threshold = non_dupli[s]

                # Change values of y to only 2 kinds of values, 0 and 1
                y = [1 if i == max(y) else 0 for i in y]

                # Get indices of the objective variable
                index_true_list = [i for i, x in enumerate(y) if x == 1]
                index_false_list = [i for i, x in enumerate(y) if x == 0]

                # Change the lists to arrays
                index_true = np.array(index_true_list)
                index_false = np.array(index_false_list)

                # Count samples whose objective variable is TRUE/FALSE
                n_more_true = np.sum(X[f, index_true] >= temporary_threshold)
                n_more_false = np.sum(X[f, index_false] >= temporary_threshold)
                n_less_true = np.sum(X[f, index_true] < temporary_threshold)
                n_less_false = np.sum(X[f, index_false] < temporary_threshold)

                # Update the information gain and the threshold
                if self._information_gain(n_more_true, n_more_false, n_less_true, n_less_false) > info_gain:
                    info_gain = self._information_gain(n_more_true, n_more_false, n_less_true, n_less_false)
                    self.threshold = temporary_threshold

        # Label
        if n_more_true >= n_more_false:
            left_label = 1
        else:
            left_label = 0

        if n_less_true >= n_less_false:
            right_label = 1
        else:
            right_label = 0

        if left_label == right_label:
            self.left_label = 1
            self.right_label = 0
        else:
            self.left_label = left_label
            self.right_label = right_label

        return self.threshold


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


    def decision_boundary(self, X, y, step=0.01, title="Decision Boundary", xlabel="1st Feature", ylabel="2nd Feature",
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
        pred = np.array(self.predict(mesh)).reshape(mesh_f0.shape)

        # Plot
        plt.figure(facecolor="azure", edgecolor="coral")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.contourf(mesh_f0, mesh_f1, pred, n_class - 1, cmap=ListedColormap(contourf_color))
        plt.contour(mesh_f0, mesh_f1, pred, n_class - 1, colors='y', linewidths=3, alpha=0.5)

        for i, target in enumerate(set(y)):
            plt.scatter(X[y == target][:, 0], X[y == target][:, 1], s=80, color=scatter_color[i], label=target_names[i],
                        marker='o')

        patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]
        plt.grid()
        plt.legend(handles=patches)
        plt.legend()
        plt.show()