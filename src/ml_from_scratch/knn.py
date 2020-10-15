import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _predict(self, x):
        # compute distances
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest
        k_indices = np.argsort(distances)[:self.k]
        # use label values
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote -- most common class label
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
