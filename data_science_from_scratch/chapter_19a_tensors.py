from typing import List, Callable

Tensor = list

def shape(tensor:Tensor) -> List[int]:
    sizes: List[int] = []

    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    
    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1,2],[2,4],[4,6]]) == [3,2]

def is1d(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list it is a higher order tensor
    Otherwise it is 1d
    """
    return not isinstance(tensor[0], list)

assert is1d([1, 2, 3]) == True
assert is1d([[1, 2], [2,4]]) == False

def tensor_sum(tensor: Tensor) -> float:
    if is1d(tensor):
        return sum(tensor)
    else:
        return sum([tensor_sum(tensor_i) for tensor_i in tensor])

assert tensor_sum([1 ,2 ,3]) == 6
assert tensor_sum([
    [
        [1, 2, 3],
        [1, 2, 3]
    ],
    [
        [1, 2, 3],
        [1, 2, 3]
    ]
 ]) == 24

def tensor_apply(f: Callable, tensor: Tensor) -> Tensor:
    if is1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]

assert tensor_apply(lambda k: k*2, [1, 2, 3]) == [2, 4, 6]
assert tensor_apply(lambda k: k*2, [[1, 2, 3], [2, 4, 6]]) == [[2, 4, 6], [4, 8, 12]]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda k: 0.0, tensor)

assert zeros_like([2, 4, 6]) == [0.0, 0.0, 0.0]
assert zeros_like([[2, 4, 6], [2, 4, 6]]) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

def tensor_combine(f: Callable, t1: Tensor, t2: Tensor) -> Tensor:
    """Apply f to corresponding elements of t1, t2"""
    if is1d(t1):
        return [f(x, y) for x,y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

import operator
assert tensor_combine(lambda j,k: j+k, [1, 2, 3], [2, 4, 6]) == [3, 6, 9]
assert tensor_combine(operator.mul, 
        [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]] ) == [[1, 4, 9], [1, 4, 9]]

