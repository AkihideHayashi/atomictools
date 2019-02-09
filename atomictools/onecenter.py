import numpy as np
from math import factorial
from scipy.special import sph_harm, eval_genlaguerre # pylint: disable=no-name-in-module

def hydrogenic_R(n: int, l: float, z: float, r: float)->float:
    return ((2**(1 + l)) * (n**(-2 - l)) * (z**(1.5 + l)) *
            np.sqrt(factorial(n - l - 1) /
            factorial(n + l)) * (r**l) * np.exp(-r * z / n) *
            eval_genlaguerre(n - l - 1, 2 * l + 1, 2 * r * z / n))

def spherical_harmonics(l, m, theta, phi):
    return sph_harm(m, l, phi, theta) * 3.5449077018105415


def hydrogenic(n, l, m, z, r, theta, phi):
    return hydrogenic_R(n, l, z, r) * spherical_harmonics(l, m, theta, phi)