# pylint: disable=E0102

import numpy as np
from numpy.linalg import norm
from numpy import eye, cos, sin, exp, log, outer, cross, arccos, arcsin
from multimethod import multimethod

from atomictools.lattice import move_to_wigner_seitz


@multimethod
def distance_matrix(pos1: np.ndarray, pos2: np.ndarray):
    assert pos1.ndim == 2
    assert pos2.ndim == 2
    return np.array([[norm(p - q) for p in pos2] for q in pos1])


@multimethod
def distance_matrix(pos1: np.ndarray, pos2: np.ndarray, ws: np.ndarray):
    assert pos1.ndim == 2
    assert pos2.ndim == 2
    def tows(r):
        move_to_wigner_seitz(r, ws)
        return r
    return np.array([[norm(tows(p - q)) for q in pos2] for p in pos1])


def cross_matrix(v):
    n = v.shape[0]
    e = np.eye(n)
    return sum(outer(cross(v, e[i]), e[i]) for i in range(n))


def center_of_gravity(coordinate, mass):
    return mass @ coordinate / np.sum(mass)


def tensor_of_inertia(coordinate, mass):
    G = mass @ coordinate / np.sum(mass)
    R = coordinate - G
    return sum((r @ r * eye(3) - outer(r, r)) * m for r, m in zip(R, mass))


def random_unit(ndim):
    """random unit vector"""
    while True:
        n = np.random.random(ndim) * 2 - 1
        if n @ n <= 1:
            return n / norm(n)


def shortest_rotation(v, n):
    """rotation matrix
    r = v x n
    theta = acos(v @ n / |v||n|)
    rodrigues_rotation(|r|, theta)
    """
    r = cross(v, n)
    r /= norm(r)
    return rodrigues_rotation(r, np.arccos(v @ n / (norm(v) * norm(n))))


@multimethod
def rodrigues_rotation(v: np.ndarray):
    t = norm(v)
    n = v / t
    return rodrigues_rotation(n, t)


@multimethod
def rodrigues_rotation(n: np.ndarray, t: float):
    return cos(t) * eye(len(n)) + (1 - cos(t)) * outer(n, n) + sin(t) * cross_matrix(n)


def random_rotate(axis=np.array([1.0, 0.0, 0.0])):
    n = random_unit(3)
    t = np.random.random() * np.pi * 2
    R1 = shortest_rotation(axis, n)
    R2 = rodrigues_rotation(n, t)
    R = R2 @ R1
    return R