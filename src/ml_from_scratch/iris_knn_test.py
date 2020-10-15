import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

## Load train / test set
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

k = KNN(k=11)
k.fit(X_train, y_train)
results = k.predict(X_test)
acc = np.sum(results == y_test) / len(y_test)
print(f"Accuracy: {acc}")
