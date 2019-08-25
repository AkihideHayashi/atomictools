import numpy as np
from scipy.optimize import minimize


def read_plumed_fes(path, fshape):
    data = np.loadtxt(f"Length_Well/LW5/Analyze/fes.dat")
    data = data.reshape((*fshape, data.shape[1])).transpose([*range(len(fshape)-1, -1, -1), -1])
    return data

def single_gaussian(x, x0, sigma, height):
    dx = (x - x0) / sigma
    dx2 = np.sum((dx * dx) / 2, axis=1)
    return np.exp(-dx2) * height

def decompose_gaussian(data, sigma, n_iter):
    gaussians = []
    n_cv = len(sigma)
    x = data[:, :n_cv]
    y = data[:, n_cv]
    yy = y[:] - 0.0
    for _ in range(n_iter):
        i = np.argmin(yy)
        new_param = np.array([*x[i, :], *sigma, yy[i] / 2])
        gaussians.append(new_param)
        new = single_gaussian(x, new_param[:n_cv], new_param[n_cv:2*n_cv], new_param[n_cv*2])
        yy -= new
    return np.array(gaussians)


def accumurate_gaussian(gs, axis):
    n = gs.shape[1]
    n_cv = (n - 1) // 2
    y = np.zeros(axis.shape[0])
    for i in range(gs.shape[0]):
        y += single_gaussian(axis, gs[i, :n_cv], gs[i, n_cv:2*n_cv], gs[i, n_cv*2])
    return y