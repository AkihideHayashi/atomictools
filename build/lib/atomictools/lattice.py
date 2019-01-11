import numpy as np


def same_vectors(ps, qs):
    for p in ps:
        for q in qs:
            if np.allclose(p, q):
                break
        else:
            return False
    return True


def insert_vector(ps, v):
    for i, p in enumerate(ps):
        if allclose(p, v):
            return
        if p[0] > v[0]:
            ps.insert(i, v)
            return
    ps.append(v)
    
    
def allclose(p, v):
    if abs(p[0] - v[0]) < 1E-8:
        if abs(p[1] - v[1]) < 1E-8:
            if abs(p[2] - v[2]) < 1E-8:
                return True
    return False


def new_points(ps):
    ret = []
    for i in range(len(ps)):
        for j in range(i+1):
            for k in range(j+1):
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        for sk in range(-1, 2):
                            new = si * ps[i] + sj * ps[j] + sk * ps[k]
                            if np.any(np.abs(new) > 1E-8):
                                insert_vector(ret, new)
    return ret


def wigner_seitz_step(ps):
    ws = []
    for i in range(len(ps)):
        wsv = True
        for w in ws:
            if np.allclose(w, ps[i]):
                wsv = False
        for j in range(len(ps)):
            if (ps[j] @ (ps[i] - ps[j])) >= 0.0 and not np.allclose(ps[i], ps[j]):
                wsv = False
        if wsv:
            insert_vector(ws, ps[i])
    return np.array(ws)


def wigner_seitz(ps):
    """Calculate lattice points which construct wigner seitz cell"""
    ws = wigner_seitz_step(new_points(ps))
    while True:
        ws_old = ws
        ws = wigner_seitz_step(new_points(ws))
        if same_vectors(ws, ws_old):
            return ws
        
def move_to_wigner_seitz(rs, ws):
    """over write r in weigner sitz cell"""
    n = np.clip(np.ceil(rs @ ws.T / np.sum(ws * ws, axis=1) - 0.5), a_max=0, a_min=None)
    return rs - n @ ws

def in_wigner_seitz(rs, ws):
    """rs in ws"""
    return np.all(rs @ ws.T <= 0.5 * np.sum(ws * ws, axis=1), axis=1)