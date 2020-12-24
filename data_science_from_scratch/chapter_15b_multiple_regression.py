from typing import TypeVar, Callable, List
import random
from chapter_5_statistics import median, standard_deviation

X = TypeVar('X')
Stat = TypeVar('Stat')

def bootstrap_sample(data: List[X]) -> List[X]:
    """randomly samples len(data) elements with replacement"""
    return  [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
        stat_function: Callable[[List[X]], Stat],
        num_samples: int) -> List[Stat]:
    """evaluates stats_fn on num_samples boostrap samples from data"""
    return  [stat_function(data) for _ in range(num_samples)]

close_to_100 = [99.5 + random.random() for _ in range(101)]
far_from_100 = ([99.5 + random.random()] +
        [random.random() for _ in range(50)] +
        [200 + random.random() for _ in range(50)])

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_far = bootstrap_statistic(far_from_100, median, 100)
