from typing import List, NamedTuple
from collections import Counter
from chapter_4_vectors import Vector, distance

def raw_majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, _ = vote_counts.most_common(1)[0]
    return winner

def majority_vote(labels: List[str]) -> str:
    """Assume labels ordered ascending distance"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values()
        if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # remove last label and re-tabulate

assert majority_vote(['a','b','c','b','a']) == 'b'

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classifier(k: int,
                   labeled_points: List[LabeledPoint],
                   new_point: Vector
                   ) -> str:
    # Order points in ascending distance
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))

    # Find labels for k closest point
    k_nearest = [lp.label for lp in by_distance[:k]]

    return majority_vote(k_nearest)
