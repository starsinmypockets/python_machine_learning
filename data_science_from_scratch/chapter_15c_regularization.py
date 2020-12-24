from book_data import inputs, daily_minutes_good
from typing import List
from chapter_4_vectors import Vector, dot, add, vector_mean
from chapter_8_gradient_descent import gradient_step
from chapter_14_linear_regression import least_squares_fit
from chapter_15_multiple_regression import sqerror_gradient, multiple_r_squared

import random

def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """gradient of the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector,
                            y: float,
                            beta: Vector,
                            alpha: float) -> Vector:
    """
    the gradient corresponding to the ith sqaured error term
    including the ridge 
    enalty
    """
    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

random.seed(0)
learning_rate = 0.001
beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0, learning_rate, 5000, 25)
print(beta_0)
assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69

beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1, learning_rate, 5000, 25)
print(beta_0_1)
assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1, learning_rate, 5000, 25)
print(beta_1)
assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good, 10, learning_rate, 5000, 25)
print(beta_10)
assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6
