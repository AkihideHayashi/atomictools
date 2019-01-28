# pylint: disable=E0102
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


@multimethod
def write_xyz(path: str, comment, symbols, positions):
    with open(path, "w") as f:
        write_xyz(f, comment, symbols, positions)
    return


@multimethod
def write_xyz(f: io.TextIOWrapper, comment, symbols, positions):
    f.write("{}\n".format(min(map(len, (symbols, positions)))))
    f.write(comment + "\n")
    for s, p in zip(symbols, positions):
        f.write("{}  {}  {}  {}\n".format(s, *fmt("{:>16.10}", p)))