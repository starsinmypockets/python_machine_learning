from chapter_4_vectors import Vector, distance
from chapter_11_machine_learning import split_data
from typing import List
import random
import tqdm
import matplotlib.pyplot as plt

dimensions = range(1,101)

def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int ) -> List[float]:
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]

random.seed(0)

avg_distances = []
min_distances = []


def curseOfDimensionality() -> None:
    for dim in tqdm.tqdm(dimensions, desc='Curse of dimensionality'):
        distances = random_distances(dim, 10000)
        avg_distances.append(sum(distances) / 10000)
        min_distances.append(min(distances))

curseOfDimensionality()
