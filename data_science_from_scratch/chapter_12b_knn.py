from typing import Tuple, Dict
from collections import defaultdict, namedtuple
from chapter_5_statistics import correlation
from chapter_11_machine_learning import split_data
from chapter_12_knn import knn_classifier, LabeledPoint
from matplotlib import pyplot as plt
import ipdb
import pprint

with open('iris.data','r') as f_open:
    _data = f_open.read()
    data = []
    # I know we should parse as csv but tough
    for row in _data.split('\n'):
        try:
            row = row.split(',')
            # each row is a tuple of a vector and a label
            data.append(LabeledPoint([float(cell) for cell in row[0:4]], row[-1]))
        except:
            print('Invalid row', row)


iris_train, iris_test = split_data(data, 0.70)
print(len(iris_train), len(iris_test), len(data))
assert int(len(data) * 0.7) == len(iris_train)

confusion_matrix: [Dict[Tuple[str,str], int]] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    try:
        predicted = knn_classifier(5, iris_train, iris.point)
        actual = iris.label
        if predicted == actual:
            num_correct += 1
        confusion_matrix[(predicted, actual)] += 1
    except:
        print('Error making prediction')

print(num_correct, f'{(num_correct / len(iris_test) * 100)}% correct')
pprint.pprint(confusion_matrix)
