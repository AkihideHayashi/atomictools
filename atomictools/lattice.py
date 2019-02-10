import numpy as np
from numpy.linalg import norm
from numpy.linalg import det


def new_points(ps):
    already = []
    for i in range(len(ps)):
        for j in range(i + 1):
            for k in range(j + 1):
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        for sk in range(-1, 2):
                            ret = si * ps[i] + sj * ps[j] + sk * ps[k]
                            if allclose(ret, [0, 0, 0]):
                                continue
                            for a in already:
                                if allclose(ret, a):
                                    break
                            else:
                                already.append(ret)
                                yield ret

def allclose(p, v):
    if abs(p[0] - v[0]) < 1E-8:
        if abs(p[1] - v[1]) < 1E-8:
            if abs(p[2] - v[2]) < 1E-8:
                return True
    return False


def sort_by_xyz(l):
    return np.array(list(map(list, sorted(map(tuple, l)))))


def primitive_cell(cell):
    new = cell.copy()
    while True:
        old = new.copy()
        cand = list(sorted(sort_by_xyz(new_points(new)), key=norm))
        new = [cand[0]]
        for c in cand[1:]:
            
            if len(new) == 1 and abs(c @ new[0] / (norm(c) * norm(new[0]))) < 0.999:
                new.append(c)
            if len(new) == 2 and det(np.array(new + [c])) > 1E-8:
                new.append(c)
                break
            else:
                continue
        if np.allclose(sort_by_xyz(new), sort_by_xyz(old)):
            return np.array(new)


def adaptable_for_wigner_seitz(r, ws):
    if allclose(r, [0, 0, 0]):
        return False
    if ws.shape[0] == 0:
        return True
    else:
        for w in ws:
            if not allclose(w, r) and w @ r >= w @ w:
                return False
        return True
        # return np.all(ws @ r <= np.sum(ws * ws, axis=1))


def wigner_seitz(lattice):
    ws = primitive_cell(lattice)
    while True:
        ws_old = ws.copy()
        cand = np.array(list(new_points(ws)))
        ws = []
        for c in cand:
            if adaptable_for_wigner_seitz(c, cand):
                ws.append(c)
        ws = sort_by_xyz(ws)
        if ws.shape == ws_old.shape and np.allclose(ws, ws_old):
            return ws
        

def move_to_wigner_seitz_matrix(rs, ws):
    """r in weigner sitz cell"""
    cont = True
    while cont:
        cont = False
        for w in ws:
            n = np.clip(np.ceil(rs @ w / (w @ w) - 0.5), a_max=None, a_min=0)
            if np.any(n > 0):
                rs -= np.outer(n, w)
                cont = True


def move_to_wigner_seitz_vector(r, ws):
    """r in weigner sitz cell"""
    cont = True
    while cont:
        cont = False
        for w in ws:
            n = np.clip(np.ceil(((w @ r) / (w @ w)) - 0.5), a_max=None, a_min=0)
            if np.any(n > 0):
                r -= n * w
                cont = True


def move_to_wigner_seitz(rs, ws):
    if rs.ndim == 1:
        move_to_wigner_seitz_vector(rs, ws)
    elif rs.ndim == 2:
        move_to_wigner_seitz_matrix(rs, ws)
    else:
        raise NotImplementedError()


def in_wigner_seitz_matrix(rs, ws):
    return np.all(rs @ ws.T <= 0.5 * np.sum(ws * ws, axis=1), axis=1)


def in_wigner_seitz_vector(r, ws):
    return np.all(ws @ r <= 0.5 * np.sum(ws * ws, axis=1))


def in_wigner_seitz(rs, ws):
    """rs in ws"""
    if rs.ndim == 1:
        return in_wigner_seitz_vector(rs, ws)
    elif rs.ndim == 2:
        return in_wigner_seitz_matrix(rs, ws)
    else:
        raise NotImplementedError()


@np.vectorize
def move_to_01(x):
    while True:
        if x < 0:
            x += 1
        elif 1 <= x:
            x -= 1
        else:
            return x


def move_to_lattice(coordinate, lattice):
    return (lattice.T @ move_to_01(np.linalg.solve(lattice.T, coordinate.T))).T



# def move_to_wigner_seitz(rs, ws):
#     """r in weigner sitz cell"""
#     n = np.clip(np.ceil(rs @ ws.T / np.sum(ws * ws, axis=1) - 0.5), a_max=0, a_min=None)
#     return rs - n @ ws

