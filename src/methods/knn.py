import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        
        test_labels = np.zeros(test_data.shape[0], dtype=int)

        if self.task_kind == "classification":
            for i in range(test_data.shape[0]):
                dists = np.linalg.norm(self.training_data - test_data[i], axis=1)
                knn_indices = np.argsort(dists)[:self.k]
                knn_labels = self.training_labels[knn_indices].astype(int)
                test_labels[i] = np.bincount(knn_labels).argmax()
        elif self.task_kind == "regression":
            for i in range(test_data.shape[0]):
                dists = np.linalg.norm(self.training_data - test_data[i], axis=1)
                knn_indices = np.argsort(dists)[:self.k]
                knn_labels = self.training_labels[knn_indices].astype(float)
                test_labels[i] = np.mean(knn_labels)
        else:
            raise ValueError(f"Unknown task kind: {self.task_kind}")
            
        return test_labels
