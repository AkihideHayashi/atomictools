import numpy as np
from scipy.optimize import root
from atomictools.unit.au import alpha


def radius_grid(g: np.ndarray, beta: float):
    """The logarithmic grid using"""
    N = len(g)
    return beta * g / (N - g)

def first_derivative_radius_grid(g: np.ndarray, r: np.ndarray, beta: float):
    """the first derivative of radius_grid dgdr"""
    N = len(g)
    return N * beta / ((r + beta) ** 2)

def second_derivative_radius_grid(g: np.ndarray, r: np.ndarray, beta: float):
    """the second derivative of radius_grid"""
    N = len(g)
    return - 2 * N * beta / ((r + beta) ** 3)

def coefficients(e: float, l: int, r: np.ndarray, vr: np.ndarray, dvrdr: np.ndarray, relativity=True):
    """coefficients for relational radial shrodinger equation
    u = rR
    -u'' - 1/(2mc^2) * dv/dr * (du/dr - u/r) + [l(l+1)/r^2 + 2m(v-e)]u = 0
    C2 u'' + C1 u' + C0 u = 0
    return np.array([C0, C1, C2])
    
    if alpha == 0: it becomes classical form
    """
    a2 = alpha * alpha if relativity else 0.0
    mr = r - 0.5 * a2 * (vr - e * r)
    mr2 = mr * 2
    relative_common = a2 / mr2 * (r * dvrdr - vr) if relativity else np.full_like(mr, 0.0)
    C2 = - r * r
    C1 = - relative_common * r
    C0 = l*(l+1) + mr2 * (vr - e * r) + relative_common
    return np.array([C0, C1, C2])

def coefficients_grid(C: np.ndarray, dxdr: np.ndarray, d2xdr2: np.ndarray):
    """transform C for coordination transform r(x)
    """
    C2 = C[2] * dxdr * dxdr
    C1 = C[2] * d2xdr2 + C[1] * dxdr
    C0 = C[0]
    return np.array([C0, C1, C2])

def integral_factor(C: np.ndarray):
    """return F
    u[i+1] = (F[1, i] * u[i] - F[0, i] * u[i-1]) / F[2, i]
    u[i-1] = (F[1, i] * u[i] - F[2, i] * u[i+1]) / F[0, i]
    """
    fm = 2 * C[2] - C[1]
    f0 = 4 * C[2] - 2 * C[0]
    fp = 2 * C[2] + C[1]
    return np.array([fm, f0, fp])

def shoot(e: float, l: int, r: np.ndarray, vr: np.ndarray, dvrdr: np.ndarray,
          dxdr: np.ndarray, d2xdr2: np.ndarray, u: np.ndarray, relativity=True):
    """shoot(e, l, r, vr, dvrdr, dxdr, d2xdr2, u)
    generate wavefunction from 0 and from inf
    overwrite u to true form
    if e is not eigenvalue, wave does not join. the discontinuity is delta
    if e is eigenvalue, delta becomes 0.
    return delta

    this routine is not compatible for positive big eigenvalue.
    """
    Cp = coefficients(e, l, r, vr, dvrdr, relativity)
    C = coefficients_grid(Cp, dxdr, d2xdr2)
    F = integral_factor(C)
    
    u[-1] = 1
    u[-2] = F[1, -1] * u[-1] / F[0, -1]
    u[0], u[1] = 0.0, 1.0
    i = C.shape[1] - 2
    if C[0, i] < 0.0:
        # e is too big
        u[:] = 0
        return 100
    while C[0, i] > 0.0:
        u[i-1] = (F[1, i] * u[i] - F[2, i] * u[i+1]) / F[0, i]
        if u[i-1] < 0:
            u[:] = 0
            return 100
        if u[i - 1] > 1e100:
            u *= 1e-100
        i -= 1
        if i == 1:
            # e is too small
            u[:] = 0
            return 100
    m = i + 1
    vi = u[m+1] - u[m-1]
    ui = u[m]
    for i in range(1, m+1):
        u[i+1] = (F[1, i] * u[i] - F[0, i] * u[i-1]) / F[2, i]
    vo = u[m+1] - u[m-1]
    uo = u[m]
    u[:m+2] *= vi / vo
    unorm = np.sqrt(np.sum(u * u / dxdr))
    u[:] /= unorm * np.sign(uo)
    return (uo * vi / vo - ui) / unorm * np.sign(vo)


def find_a_eigensystem(e0: float, l: int, r: np.ndarray,
                       vr: np.ndarray, dvrdr: np.ndarray,
                       dxdr: np.ndarray, d2xdr2: np.ndarray, u: np.ndarray, relativity=True):
    """returns eigenvalue
    overwrite u by eigenfunction
    """
    def inner(e):
        return shoot(e, l, r, vr, dvrdr, dxdr, d2xdr2, u, relativity)
    return root(inner, e0)


def find_cutoff(u: np.ndarray, tol: float):
    """The last i which abs(u[i]) >= tol
    """
    for i in range(len(u)-1, -1, -1):
        if abs(u[i]) > tol:
            return i + 1