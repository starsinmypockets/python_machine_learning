from typing import List, Tuple, Callable

Matrix = List[List[float]]
Vector = List[float]

def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

assert shape([[1,2,3], [1,2,3]]) == (2,3)

def get_row(A: Matrix, i: int) -> Vector:
    return A[i]

assert get_row([[1,2,3], [2,3,4]], 1) == [2,3,4]

def get_col(A: Matrix, i: int) -> Vector:
    return [a[i] for a in A]

assert get_col([[2,4,6], [2,4,6], [2,4,6]], 1) == [4,4,4]

def make_matrix(num_rows: int, num_cols: int, entry_fn:Callable) -> Matrix:
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int):
    return make_matrix(n, n, lambda i,j: 1 if i == j else 0)

assert identity_matrix(8) == [[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]]
