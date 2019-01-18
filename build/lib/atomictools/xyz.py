import numpy as np
from multimethod import multimethod
import io

from atomictools.tools import contract, expand, read_matrix, skip_until, read_aslongas, fmt


@multimethod
def read_xyz(f: io.TextIOWrapper):
    n = int(next(f))
    comment = next(f)
    m = read_matrix(f, n)
    symbols = m[:, 0]
    positions = m[:, 1:4].astype(np.float64)
    return comment, symbols, positions

@multimethod
def read_xyz(path: str):
    with open(path) as f:
        return read_xyz(f)
    return
