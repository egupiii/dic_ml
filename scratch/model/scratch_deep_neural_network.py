from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy



class ScratchDeepNeuralNetrowkClassifier:
    """
    Implement neural network classifier.

    Parameters
    ----------
    num_epoch : int
        Number of epochs

    batch_size : int
        Size of batch

    verbose : bool
        True if outputting learning process


    Attributes
    ----------
    loss : list
        List of arrays of records of loss on train dataset

    val_loss : list
        List of arrays of records of loss on validation dataset

    layers : list
        List of layers
    """

    def __init__(self, num_epoch, batch_size, verbose=True):
        # Record hyperparameters as attribute
        self.epoch = num_epoch
        self.batch_size = batch_size
        self.verbose = verbose

        # Prepare lists for arrays to record losses
        self.loss = []
        self.val_loss = []
        # Prepare lists for arrays to record losses
        self.layers = []


    def add(self, layer):
        self.layers += [layer]


    def forward_layer(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X


    def backward_layer(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y)

        return y


    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit neural network classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of train dataset

        y : ndarray, shape (n_samples, )
            Correct values of train dataset

        X_val : ndarray, shape (n_samples, n_features)
            Features of validation dataset

        y_val : ndarray, shape (n_samples, )
            Correct values of validation dataset
        """

        if self.verbose:
            count = 0

        for i in range(self.epoch):
            # Initialize
            get_mini_batch = GetMiniBatch(X, y, batch_size=self.batch_size)

            if (X_val is not None) and (y_val is not None):
                get_mini_batch_val = GetMiniBatch(X_val, y_val, batch_size=self.batch_size)

                for ((mini_X_train, mini_y_train), (mini_X_val_train, mini_y_val_train)) in zip(get_mini_batch,
                                                                                                get_mini_batch_val):
                    # Forwardpropagation per iteration
                    Z3 = self.forward_layer(mini_X_train)
                    Z3_val = self.forward_layer(mini_X_val_train)

                    # Loss
                    if self.verbose:
                        # Initialize
                        loss = Loss()
                        # Compute losses
                        L = loss.cross_entropy_loss(mini_y_train, Z3)
                        L_val = loss.cross_entropy_loss(mini_y_val_train, Z3_val)

                    # Backforwardpropagation per iteration
                    dX = self.backward_layer(mini_y_train)
                    dX_val = self.backward_layer(mini_y_val_train)


            else:
                for mini_X_train, mini_y_train in get_mini_batch:
                    # Forwardpropagation per iteration
                    Z3 = self.forward_layer(mini_X_train)

                    # Loss
                    if self.verbose:
                        # Initialize
                        loss = Loss()
                        # Compute losses
                        L = loss.cross_entropy_loss(mini_y_train, Z3)

                    # Backforwardpropagation per iteration
                    dX = self.backward_layer(mini_y_train)

            # Output learning process if verbose is True
            if self.verbose:
                self.loss += [sum(L) / self.batch_size]
                if (X_val is not None) and (y_val is not None):
                    self.val_loss += [sum(L_val) / self.batch_size]
                    print("{0}th loss: {1}, val_loss: {2}".format(count + 1, self.loss[count], self.val_loss[count]))
                else:
                    print(self.loss[count])
                count += 1


    def predict(self, X):
        """
        Predict by neural network classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Samples


        Returns
        -------
        ndarray, shape (n_samples, 1)
            Results of prediction
        """

        Z3 = self.forward_layer(X)

        return np.argmax(Z3, axis=1)


    def plot_learning_record(self):
        """
        Plot learning records.
        """

        plt.figure(facecolor="azure", edgecolor="coral")

        plt.plot(self.loss, label="loss")
        plt.plot(self.val_loss, label="val_loss")

        plt.title("Learning Records")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.legend()
        plt.show()


    def compute_index_values(self, y, y_pred):
        """
        Compute Index values.

        Parameters
        ----------
        X: ndarray, shape(n_samples,n_features)
            Features of train dataset

        y: ndarray, shape(n_samples,)
            Correct values of train dataset
        """

        print("accuracy score:", accuracy_score(y, y_pred))


    def plot_misclassification(self, X_val, y_val, y_pred):
        """
        Plot results of misclassification. Show "Results of prediction/Corrects" above images.

        Parameters
        ----------
        y_pred : ndarray, shape (n_samples,)
            Results of prediction

        y_val : ndarray, shape (n_samples,)
            Correct labels of validation data

        X_val : ndarray, shape (n_samples, n_features)
            Features of validation data
        """

        # Number of results I want to plot
        num = 36

        true_false = y_pred == y_val
        false_list = np.where(true_false == False)[0].astype(np.int)

        if false_list.shape[0] < num:
            num = false_list.shape[0]
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=0.8, bottom=0, top=0.8, hspace=1, wspace=0.5)
        for i in range(num):
            ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
            ax.set_title("{} / {}".format(y_pred[false_list[i]], y_val[false_list[i]]))
            ax.imshow(X_val.reshape(-1, 28, 28)[false_list[i]], cmap='gray')



class FC:
    """
    Fully connected layer from a layer of n_nodes1 to a layer of n_nodes2

    Parameters
    ----------
    n_nodes1 : int
        Number of nodes of the previous layer

    n_nodes2 : int
        Number of nodes of the following layer

    initializer : Instance
        Instance of initialization method

    optimizer : Instance
        Instance of optimisation method


    Attributes
    ----------
    W : ndarray, shape (n_nodes1, n_nodes2)
        Weight

    B : ndarray, shape (n_nodes2,)
        Bias

    dW : float
        Gradient of weight

    dB : float
        Gradient of bias
    """

    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.initializer = initializer
        self.optimizer = optimizer

        # Initialize self.W and self.B by using initializer method
        self.W = self.initializer.W(self.n_nodes1, self.n_nodes2)
        self.B = self.initializer.B(self.n_nodes2)

        self.dW = 0
        self.dB = 0


    def forward(self, X):
        """
        Forwardpropagation

        Parameters
        ----------
        X : ndarray, shape (batch_size, n_nodes1)
            Input


        Returns
        ----------
        ndarray, shape (batch_size, n_nodes2)
            Output
        """

        self.Z = copy.deepcopy(X)

        return np.dot(X, self.W) + self.B


    def backward(self, dA):
        """
        Backwardpropagation

        Parameters
        ----------
        dA : ndarray, shape (batch_size, n_nodes2)
            Gradient given from the following layer


        Returns
        ----------
        dZ : ndarray, shape (batch_size, n_nodes1)
            Gradient given to the next layer
        """

        self.dB = np.average(dA)
        self.dW = np.dot(self.Z.T, dA) / dA.shape[0]

        dZ = np.dot(dA, self.W.T)

        # Update
        self = self.optimizer.update(self)

        return dZ



class SimpleInitializer:
    """
    Simple initialization by Gaussian distribution

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian distribution
    """

    def __init__(self, sigma):
        self.sigma = sigma


    def W(self, n_nodes1, n_nodes2):
        """
        Initialize a weight.

        Parameters
        ----------
        n_nodes1 : int
            Number of nodes of the previous layer

        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        W : ndarray, shape (n_nodes1, n_nodes2)
            Weight
        """

        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)

        return W.astype("f")


    def B(self, n_nodes2):
        """
        Initialize a bias.

        Parameters
        ----------
        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        B : ndarray, shape (n_nodes2,)
            Bias
        """

        B = self.sigma * np.random.randn(1, n_nodes2)

        return B.astype("f")



class SGD:
    """
    Stochastic Gradient Descent

    Parameters
    ----------
    lr : float
        Learning rate
    """

    def __init__(self, lr):
        self.lr = lr


    def update(self, layer):
        """
        Update weights and biases of layers.

        Parameters
        ----------
        layer : Instance
            Instance of preupdated layer


        Returns
        ----------
        layer : Instance
            Instance of updated layer
        """

        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB

        return layer



class Sigmoid:
    """
    Sigmoid function
    """

    def forward(self, A):
        """
        Forwardpropagation

        Parameters
        ----------
        A : ndarray, shape (batch_size,)
            Vector from the previous layer of kth class


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        self.A = A

        return 1 / (1 + np.exp(-self.A))


    def backward(self, dA):
        """
        Backpropagation

        Paramaters
        ----------
        dA : ndarray, shape (batch_size, n_nodes2)
            Gradient given from the following layer


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        Z = self.forward(self.A)

        return Z * (1 - Z) * dA



class Tanh:
    """
    tanh function
    """

    def forward(self, A):
        """
        Forwardpropagation

        Parameters
        ----------
        A : ndarray, shape (batch_size,)
            Vector from the previous layer of kth class


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        self.A = A

        return np.tanh(self.A)


    def backward(self, dA):
        """
        Backpropagation

        Parameters
        ----------
        dA : ndarray, shape (batch_size, n_nodes2)
            Gradient given from the following layer


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        Z = self.forward(self.A)

        return (1 - Z ** 2) * dA



class Softmax:
    """
    Softmax function

    Attributes
    ----------
    Z : ndarray, shape (batch_size, ith n_nodes)
        Output
    """

    def __init__(self):
        self.Z = None


    def forward(self, A):
        """
        Forwardpropagation

        Parameters
        ----------
        A : ndarray, shape (batch_size,)
            Vector from the previous layer of kth class


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        A -= np.max(A)

        Z = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)

        self.Z = Z

        return Z


    def backward(self, y):
        """
        Backwardpropagation

        Parameters
        ----------
        y : ndarray, shape (n_samples, 1)
            Correct values


        Returns
        -------
        ndarray, shape (batch_size,)
            Probability vector of kth class
        """

        return self.Z - y



class Relu:
    """
    ReLU function
    """

    def forward(self, A):
        """
        Forwardpropagation

        Parameters
        ----------
        A : ndarray, shape (batch_size,)
            Vector from the previous layer of kth class


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        self.A = A

        return np.where(self.A <= 0, 0, self.A)


    def backward(self, dA):
        """
        Backpropagation

        Parameters
        ----------
        dA : ndarray, shape (batch_size, n_nodes2)
            Gradient given from the following layer


        Returns
        -------
        ndarray, shape (batch_size, ith n_nodes)
            Output
        """

        return np.where(self.A <= 0, 0, 1) * dA



class XavierInitializer:
    """
    Initialize a weight by Xavier's method, and initialize a bias.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian distribution
    """

    def __init__(self, sigma):
        self.sigma = sigma


    def W(self, n_nodes1, n_nodes2):
        """
        Initialize a weight by Xavier's method.

        Parameters
        ----------
        n_nodes1 : int
            Number of nodes of the previous layer

        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        W : ndarray, shape (n_nodes1, n_nodes2)
            Weight
        """

        W = self.sigma * np.random.randn(n_nodes1, n_nodes2) / np.sqrt(n_nodes1)

        return W.astype("f")


    def B(self, n_nodes2):
        """
        Initialize a bias.

        Parameters
        ----------
        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        B : ndarray, shape (n_nodes2,)
            Bias
        """

        B = self.sigma * np.random.randn(1, n_nodes2)

        return B.astype("f")



class HeInitializer:
    """
    Initialize a weight by He's method, and initialize a bias.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian distribution
    """

    def __init__(self, sigma):
        self.sigma = sigma


    def W(self, n_nodes1, n_nodes2):
        """
        Initialize a weight by Xavier's method.

        Parameters
        ----------
        n_nodes1 : int
            Number of nodes of the previous layer

        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        W : ndarray, shape (n_nodes1, n_nodes2)
            Weight
        """

        W = self.sigma * np.random.randn(n_nodes1, n_nodes2) / np.sqrt(2 / n_nodes1)

        return W.astype("f")


    def B(self, n_nodes2):
        """
        Initialize a bias.

        Parameters
        ----------
        n_nodes2 : int
            Number of nodes of the following layer


        Returns
        ----------
        B : ndarray, shape (n_nodes2,)
            Bias
        """

        B = self.sigma * np.random.randn(1, n_nodes2)

        return B.astype("f")



class AdaGrad:
    """
    AdaGrad

    Parameters
    ----------
    lr : float
        Learning rate

    Attributes
    ----------
    h : float
        Sum of squares of all gradients up to the previous iterations about ith layer
    """

    def __init__(self, lr):
        self.lr = lr

        self.h = 0


    def update(self, layer):
        """
        Update weights and biases of layers.

        Parameters
        ----------
        layer : Instance
            Instance of preupdated layer


        Returns
        ----------
        layer : Instance
            Instance of updated layer
        """

        self.h += layer.dW * layer.dW

        layer.W -= self.lr * layer.dW / np.sqrt(self.h + 1e-7)
        layer.B -= self.lr * layer.dB

        return layer



class Loss:
    """
    Compute loss.
    """

    def cross_entropy_loss(self, y, y_pred):
        """
        Cross entropy error

        Parameters
        ----------
        y : ndarray, shape (n_samples, 1)
            Correct values

        y_pred : ndarray, shape (n_samples, 1)
            Predicted values


        Returns
        -------
        ndarray, shape (n_samples, 1)
            Cross entropy error
        """

        return np.sum(-1 * y * np.log(y_pred + 1e-10), axis=1)



class GetMiniBatch():
    """
    Iterator to get a mini-batch

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      Train dataset

    y : ndarray, shape (n_samples, 1)
      Correct values

    batch_size : int
      Size of batch

    seed : int
      Seed of random numbers of Numpy
    """

    def __init__(self, X, y, batch_size=10, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0] / self.batch_size).astype(np.int)


    def __len__(self):
        return self._stop


    def __getitem__(self, item):
        p0 = item * self.batch_size
        p1 = item * self.batch_size + self.batch_size

        return self.X[p0:p1], self.y[p0:p1]


    def __iter__(self):
        self._counter = 0

        return self


    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()

        p0 = self._counter * self.batch_size
        p1 = self._counter * self.batch_size + self.batch_size

        self._counter += 1

        return self.X[p0:p1], self.y[p0:p1]



class Dropout:
    """
    Dropout

    Parameters
    ----------
    dropout_ratio : float
        Ratio of dropout


    Attributes
    ----------
    mask : float
        Mask
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio

        self.mask = None


    def forward(self, X, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*X.shape) > self.dropout_ratio
            return X * self.mask
        else:
            return X * (1 - self.dropout_ratio)


    def backward(self, dA):
        return dA * self.mask