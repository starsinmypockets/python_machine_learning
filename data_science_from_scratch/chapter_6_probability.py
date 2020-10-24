import math

SQRT_2_PI = math.sqrt(2*math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_2_PI * sigma))

def normal_cdf(x: float, mu: float=0, sigma:float=1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float,
                       mu: float=0,
                       sigma: float=1,
                       tolerance: float=.00001) -> float:
    """find appropriate inverse using binary search"""
    # only compute standard:
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10.0

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:    # too low search above
            low_z = mid_z
        else:            # too hi search below
            hi_z = mid_z
    return mid_z

def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))
