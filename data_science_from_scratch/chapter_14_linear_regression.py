from chapter_4_vectors import Vector
from chapter_5_statistics import correlation, standard_deviation, mean
from chapter_11_machine_learning import split_data
from chapter_12_knn import LabeledPoint
from typing import Tuple

def predict(alpha: float, beta: float, x_i: float ) -> float:
    return beta * x_i + alpha

assert predict(2,4,6) == 26

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """The error when predicting beta * x_i + alpha
    when the acutal value is y_i"""
    return predict(alpha, beta, x_i) - y_i

assert error(2,4,6,10) == 16

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """Given two vectors x and y find the least squares value of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

assert least_squares_fit(x,y) == (-5, 3)

with open('iris.data','r') as f_open:
    _data = f_open.read()
    data = []
    # I know we should parse as csv but tough
    for row in _data.split('\n'):
        try:
            row = row.split(',')
            # each row is a tuple of a vector and a label
            print(row)
            data.append(LabeledPoint([float(cell) for cell in row[0:4]], row[-1]))
        except:
            print('Invalid row', row)

sepal_length = [row[0][0] for row in data if row[1] == 'Iris-versicolor']
sepal_length_train, sepal_length_test = split_data(sepal_length, 0.7)
petal_length = [row[0][1] for row in data if row[1] == 'Iris-versicolor']
petal_length_train, petal_length_test = split_data(petal_length, 0.7)

fit = least_squares_fit(sepal_length, petal_length)
predicted = [round(y * fit[1] + fit[0], 1) for y in sepal_length_test]
errors = [abs((a[0] - a[1]) / a[0]) for a in zip(petal_length_test, predicted)]

print(data[0])
print('sepal_lengths:')
print(sepal_length)
print('petal lengths:')
print(petal_length)
print('predicted')
print(predicted)
print('fit -->', least_squares_fit(sepal_length, petal_length))
print('prediction error')
print(errors)
