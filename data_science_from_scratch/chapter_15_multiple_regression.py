from typing import List
from chapter_4_vectors import Vector, dot, vector_mean
from chapter_8_gradient_descent import gradient_step
from book_data import daily_minutes_good, inputs
import random, tqdm

def predict(x: Vector, beta: Vector) -> float:
   """assumes that the first el of x is 1"""
   return dot(x, beta)

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> Vector:
    return error(x, y, beta) ** 2

x = [1, 2, 3]
y = 30
beta = [4, 4, 4]

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    """
    Find beta that minimizes sum of squared errors
    assuming the model y = dot(x, beta)
    """
    # start with random guess
    guess = [random.random() for _ in xs[0]]
    for _ in tqdm.trange(num_steps, desc="Least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                    for x, y in zip(batch_xs, batch_ys)])

            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

# now we'll use the implementation to look at the friend data from the book

random.seed(0)
beta = least_squares_fit(inputs, daily_minutes_good)

assert  30.5 < beta[0] <  30.7
assert  0.96 < beta[1] <  1.00
assert -1.78 < beta[2] < -1.77
assert  0.67 < beta[3] <  0.68
