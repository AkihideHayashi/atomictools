import numpy as np
from functools import wraps


def expand(num, val):
    return np.array([v for n, v in zip(num, val) for _ in range(n)])

def contract(vals):
    num = [0]
    val = [vals[0]]
    for v in vals:
        if v == val[-1]:
            num[-1] += 1
        else:
            val.append(v)
            num.append(1)
    return np.array(num), np.array(val)

def read_matrix(f, n):
    lines = [next(f).split() for _ in range(n)]
    return np.array(lines)


def skip_until(f, val):
    for line in f:
        if val in line:
            return line
        
        
def read_aslongas(f, val):
    lines = []
    for line in f:
        if val in line:
            lines.append(line)
        else:
            return lines
        
@np.vectorize
def fmt(f, x):
    return f.format(x)


def withopen(func):
    @wraps(func)
    def wrapper(path, *args, **kwargs):
        if isinstance(path, str):
            with open(path) as f:
                return func(f, *args, **kwargs)
        else:
            return func(f, *args, **kwargs)
    return wrapper