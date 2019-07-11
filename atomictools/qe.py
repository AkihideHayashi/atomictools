import io
import re
import numpy as np
from multimethod import multimethod

re_atomic_position = re.compile(r"ATOMIC_POSITIONS (.*)\n")
re_lattice_parameter = re.compile(r"\s*lattice parameter \(alat\)  =\s*(.*)\s*a\.u\.")
re_crystal_axes = re.compile(r"\s*a\(.*\) = \(\s*(.*)\)\s*\n")

def to_bohr(alat, lattice, kind, element, position):
    if kind == '(crystal)':
        return alat * lattice, element, alat * position @ lattice
    elif kind == '(bohr)':
        return alat * lattice, element, position
    else:
        raise NotImplementedError(kind)


@multimethod
def read_qe_out(path: str):
    with open(path) as f:
        return read_qe_out(f)

@multimethod
def read_qe_out(f: io.TextIOWrapper):
    n = 0
    axes = []
    kinds = []
    elements = []
    positions = []
    lattice = []
    for line in f:
        if "number of atoms/cell      =" in line:
            n = int(line.split('=')[-1])
        if "lattice parameter (alat)" in line:
            alat = float(re_lattice_parameter.match(line).groups()[0])
        if "new lattice vectors" in line:
            raise NotImplementedError()
        if "crystal axes" in line:
            for _ in range(3):
                axes.append(re_crystal_axes.match(next(f))[1].split())
        if "ATOMIC_POSITIONS" in line:
            ap = re_atomic_position.match(line).groups()[0]
            nr = np.array([next(f).split() for _ in range(n)])
            kinds.append(ap)
            elements.append(nr[:, 0])
            positions.append(nr[:, 1:4].astype(np.float64))
    initial_axes = np.array(axes, dtype=np.float64)
    if not lattice:
        lattice = [initial_axes for _ in range(len(positions))]
    lattice, element, position = zip(*[to_bohr(alat, l, k, e, p) for l, k, e, p in zip(lattice, kinds, elements, positions)])
    return alat, lattice, element, position


@multimethod
def read_pos(path: str, n: int):
    with open(path) as f:
        return read_pos(f, n)

@multimethod
def read_pos(f: io.TextIOWrapper, n: int):
    steps = []
    poses = []
    while True:
        try:
            step, pos = read_pos_atoms(f, n)
            steps.append(step)
            poses.append(pos)
        except StopIteration:
            break
    return steps, np.array(poses)

def read_pos_atoms(f, n):
    step, _ = map(float, next(f).split())
    pos = [[float(w) for w in next(f).split()] for _ in range(n)]
    return step, np.array(pos)


@multimethod
def read_evp(path: str):
    with open(path) as f:
        return read_evp(f)

@multimethod
def read_evp(f: io.TextIOWrapper):
    title = np.array(next(f).split()[1:])
    evp = [[float(w) for w in line.split()] for line in f if line.strip()]
    return title, np.array(evp)
