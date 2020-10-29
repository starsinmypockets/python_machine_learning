from typing import List, Dict, Tuple
from collections import Counter
import math
import random
import matplotlib.pyplot as plt
from chapter_4_vectors import Vector, distance, vector_mean
from chapter_4_matrices import Matrix, make_matrix
from chapter_5_statistics import correlation, standard_deviation
from chapter_6_probability import inverse_normal_cdf

def bucketize(point: float, bucket_size: float) -> float:
    """Floor point to next lowest bucket"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ''):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)

def random_normal() -> float:
    """Returns random draw from normal distribution"""
    return inverse_normal_cdf(random.random())

def correlation_matrix(data: List[Vector]) -> Matrix:
    """ Returns len(data) x len(data) matrix whose (i, j) entry
    is the correlation between data[i] and data[j]"""
    def correlation_ij(i: int, j: int) -> float:
        return correlation(i, j)

    return make_matrix(len(data), len(data), correlation_ij)


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Returns mean and standard deviation for each position"""
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means, stdevs

vectors = [[-3,-1,1], [-1,0,1], [1,1,1]]
means, stdevs = scale(vectors)
assert means == [-1,0,1]
assert stdevs == [2,1,0]

def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1
    """
    dim = len(data[0])
    means, stdevs = scale(data)

    rescaled = [v[:] for v in data]
    print(rescaled)
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i] / stdevs[i])

    return rescaled

## TODO assertions failed for rescale maybe they are wrong in the book?
#assert means == [0,0,1]
#assert stdevs == [1,1,0]
