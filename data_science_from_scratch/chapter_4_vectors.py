import math
from typing import List

Vector = List[float]

def add(v: Vector, w:Vector):
    """Add two vectors"""
    assert len(v) == len(w)
    return [v[x] + w[x] for x in range(len(v))]

assert add([1,2,3], [4,5,6]) == [5,7,9]

def subtract(v: Vector, w:Vector):
    """Subtract two vectors"""
    assert len(v) == len(w)
    return [v[x] - w[x] for x in range(len(v))]

assert subtract([3,2,1],[1,1,1]) == [2,1,0]

def vector_sum(vs: List[Vector]):
    """Sum a list of vectors"""
    assert vs, "No vectors provided!"
    num_els = len(vs[0])
    assert all(len(v) == num_els for v in vs), "Vectors must be the same size"
    return [sum(v[i] for v in vs)
           for i in range(num_els)]

assert vector_sum([[1,2], [3,4], [5,6]]) == [9,12]

def scalar_multiply(c: float, v: Vector):
    """Multiply ever element in a vector by a factor"""
    return [c*x for x in v]

assert scalar_multiply(3, [1,2,3]) == [3,6,9]

def vector_mean(vs: List[Vector]):
    """Compute element-wise average"""
    n = len(vs)
    return scalar_multiply(1/n, vector_sum(vs))

assert vector_mean([[1,2], [3,4], [5,6]]) == [3,4]

def dot(v: Vector, w: Vector):
    """v1 * w1 + ... vn * w*n"""
    return sum([v_i * w_i for v_i,w_i in zip(v,w)])

assert dot([1,2,3],[1,2,3]) == 14

def sum_of_squares(v):
    """v1 * v1 + ... vn * vn"""
    return dot(v,v)

assert sum_of_squares([3,6,9]) == sum([9, 36, 81])

def magnitude(v: Vector):
    """Return magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3,4]) == 5

def squared_distance(v: Vector, w:Vector):
    """(v1 - w1)**2 + ... (vn - wn)**2"""
    return sum_of_squares(subtract(v,w))

def def distance(v: Vector, w: Vector):
    """ Compute distance between v and w"""
    return magnitude(subtract(v,w))
