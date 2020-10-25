from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt
from chapter_4_vectors import Vector
from chapter_4_matrices import Matrix, make_matrix
from chapter_5_statistics import correlation
from chapter_6_probability import inverse_normal_cdf
import random

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
