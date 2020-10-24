from typing import Callable, TypeVar, List, Iterator
from chapter_4_vectors import Vector, dot
import random
from chapter_4_vectors import distance, add, scalar_multiply, vector_mean

def square(x:float) -> float:
    return x * x

def sum_of_squares(v: Vector) -> float:
    return  dot(v)

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
      return (f(x+h) - f(x)) / h

def derivative(x: float) -> float:
    return 2 * x

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Return ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)
        for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f,v,i,h)
           for i in range(len(v))]

def gradient_step(v: Vector,
                  gradient: Vector,
                  step_size: float) -> Vector:
    """Moves `step_size` in the `gradient_vector` by `step_size`"""
    assert(len(gradient) == len(v))
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
     return

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

T = TypeVar('T')

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """generates `batch-size`d batches from `dataset`"""
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
