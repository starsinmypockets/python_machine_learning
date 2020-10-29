from typing import List
from collections import Counter
from chapter_4_vectors import sum_of_squares, dot
import math

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

assert mean([2,4,6,8]) == 20 / 4

def _median_even(xs: List[float]) -> float:
    return (sorted(xs)[(len(xs) // 2) -1] + sorted(xs)[len(xs) // 2]) / 2

def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]

def median(xs: List[float]) -> float:
    return _median_even(xs) if len(xs) % 2 == 0 else _median_odd(xs)

assert median([2, 4, 6, 8]) == (4 + 6) / 2
assert median([2, 4, 6, 8, 10]) == 6

def quantile(xs: List[float], p: float) -> float:
    ''''returns the pth-percentile of list xs'''
    return sorted(xs)[int(p * len(xs))]

assert quantile([10, 30, 40, 40, 50, 60, 70, 80, 90, 100], .5) == 60
assert quantile([10, 30, 40, 40, 50, 60, 70, 80, 90, 100], .4) == 50

def mode(xs: List[float]) -> List[float]:
    counts = Counter(xs)
    _max = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == _max]

assert mode([2,4,4,6,8]) == [4]
assert mode([2,4,4,6,6,8]) == [4,6]

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

assert data_range([2,4,8]) == 8 - 2
assert data_range([2.2,4.4,8.8]) == 8.8 - 2.2
assert data_range([2,2,2]) == 0

def de_mean(xs: List[float]) -> List[float]:
    '''translate xs by subtracting its mean, so resultant mean is 0'''
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    '''Almost avg square deviation from mean'''
    assert len(xs) > 1, "variance requires list of at least two elements"
    n = len(xs)
    dev = de_mean(xs)
    return sum_of_squares(dev) / (n - 1)

## tbh I'm not sure what assertion this should pass
# print('variance', variance([2,4,6,8,10]))

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

# print('Standard deviation', standard_deviation([2,4,6,8,10]))

def interquartile(xs: List[float]) -> float:
    '''Returns difference between 75th and 25th percentile'''
    return quantile(xs, .75) - quantile(xs, .25)

# print(quantile([1,2,3,4], .75), quantile([1,2,3,4], .25))
# print(interquartile([1,2,3,4]))

def covariance(xs: List[float], ys: List[float]) -> List[float]:
    assert len(xs) == len(ys), 'Lists must be of equal length'
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

# print(covariance([1,2,3,4,12],[2,4,7,18,33]))
# print(covariance([1,2,3,4], [1,2,3,4]))
# print(covariance([1,4,16,32], [1,1,1,1]))

def correlation(xs: List[float], ys:List[float]) -> float:
    '''Measure how much xs, ys vary from their means'''
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0

## I'm not sure how to calculate assertions
# print(correlation([1,2,3,4,12],[2,4,7,18,33]))
# print(correlation([1,2,3,4], [20,12,3,4]))
# print(correlation([-2,-1,0,1,2], [2,1,0,1,2]))
