from typing import Tuple
from chapter_6_probability import normal_cdf, inverse_normal_cdf
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a binomial n,p"""
    mu = p* n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

normal_probability_below = normal_cdf

# It's above the threshhold if it's not below
def normal_probability_above(lo: float,
                             mu: float=0,
                             sigma: float=1) -> float:
    """Probability that N(mu, sigma) is greater than lo"""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it is less than hi but not below lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    """The probability that N(mu, sigma) is between hi and lo"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's ouside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    """The probability that N(mu, sigma) is not between hi and lo"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float=0,
                            sigma: float=1) -> Tuple[float, float]:
    """Returns the symmetric (about the mean) bounds that contain
    the specified probability"""
    tail_probability = (1 - probability) / 2
    inverse_normal_cdf(1 - probability, mu, sigma)
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound
