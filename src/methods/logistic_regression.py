import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, break_threshold=1e-6, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            break_threshold (float): threshold for breaking the training loop
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.break_threshold = break_threshold
        self.max_iters = max_iters

        self.W = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """

        y = label_to_onehot(training_labels)

        W = np.zeros((training_data.shape[1], y.shape[1]))

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def grad_loss_fn(X, y, y_pred):
            return X.T @ (y_pred - y)
        
        for i in range(self.max_iters):
            y_pred = sigmoid(W.T @ training_data.T).T
            dW = self.lr * grad_loss_fn(training_data, y, y_pred)
            W -= dW
            if np.linalg.norm(dW) < self.break_threshold:
                break

        self.W = W

        pred_labels = np.argmax(sigmoid(W.T @ training_data.T), axis=0)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        pred_labels = np.argmax(sigmoid(self.W.T @ test_data.T), axis=0)

        return pred_labels
