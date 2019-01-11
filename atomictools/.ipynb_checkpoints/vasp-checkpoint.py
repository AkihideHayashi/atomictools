from atomictools.tools import contract, expand, read_matrix, skip_until, read_aslongas, fmt

import numpy as np
from multimethod import multimethod
import io


@np.vectorize
def bool_to_TF(x):
    return str(x)[0]


def read_poscar_switch(f):
    line = next(f).strip()
    if line[0] in ["S", "s"]:
        selective = True
        line = next(f).strip()
    else:
        selective = False
    if line[0] in ["C", "c", "K", "k"]:
        cartesian = True
    else:
        cartesian = False
    return selective, cartesian


@multimethod
def read_poscar(f: io.TextIOWrapper):
    system = next(f).strip()
    unit = float(next(f).strip())
    cell = read_matrix(f, 3).astype(np.float64)
    sym = next(f).split()
    num = [int(w) for w in next(f).split()]
    symbols = expand(num, sym)
    selective, cartesian = read_poscar_switch(f)
    ct = read_matrix(f, len(symbols))
    if cartesian:
        coordinate = ct[:, 0:3].astype(np.float64)
    else:
        coordinate = ct[:, 0:3].astype(np.float64) @ cell
    if selective:
        selectived = np.logical_or(ct[:, 3:6] == "T", ct[:, 3:6] == "t")
    else:
        selectived = np.full((len(symbols), 3), True)
    return system, cell * unit, symbols, coordinate * unit, selectived


@multimethod
def read_poscar(path: str):
    with open(path) as f:
        return read_poscar(f)
    return


@multimethod    
def write_poscar(f: io.TextIOWrapper, lattice, symbols, positions, selective, title, cart, unit):
    f.write(f"{title}\n")
    f.write(f"{unit}\n")
    for l in lattice / unit:
        f.write(f"{l[0]:<016.10}  {l[1]:<016.10}  {l[2]:<016.10}\n")
    n, s = contract(symbols)
    f.write("  ".join(s) + "\n")
    f.write("  ".join(map(str, n)) + "\n")
    if selective is not None:
        f.write("Selective Dynamics\n")
        sel = bool_to_TF(selective)
    else:
        sel = None
    if cart:
        f.write("Cartesian\n")
        pos = fmt("{:<016.10}", positions / unit)
    else:
        f.write("Direct\n")
        pos = fmt("{:<016.10}", np.linalg.solve(lattice.T, positions.T).T)
    if selective is not None:
        for p, s in zip(pos, sel):
            f.write("  ".join(p) + "    " + "  ".join(s) + "\n")
    else:
        for p in pos:
            f.write("  ".join(p) + "\n")
    return


@multimethod
def write_poscar(path: str, lattice, symbols, positions, selective, title, cart, unit):
    with open(path, "w") as f:
        write_poscar(f, lattice, symbols, positions, selective, title, cart, unit)
    return


def read_outcar_a_position_force(f, n):
    skip_until(f, "POSITION                                       TOTAL-FORCE (eV/Angst)")
    next(f)
    pf = read_matrix(f, n)
    return pf[:, :3].astype(np.float64), pf[:, 3:6].astype(np.float64)


def read_outcar_a_energy(f):
    return float(skip_until(f, "energy  without entropy=").split("=")[2])


def read_outcar_a_lattice(f):
    skip_until(f, "VOLUME and BASIS-vectors are now :")
    next(f)
    next(f)
    next(f)
    next(f)
    lattices = read_matrix(f, 3)
    return lattices[:, 0:3].astype(np.float64)


@multimethod
def read_outcar_trajectory(f: io.TextIOWrapper):
    """returns positions, forces, energies"""
    skip_until(f, "INCAR:")
    s = [line.split()[2] for line in read_aslongas(f, "POTCAR:")[:-1]]
    n = [int(w) for w in skip_until(f, "ions per type =").split("=")[1].split()]
    symbols = expand(n, s)
    n = len(symbols)
    energies = []
    positions = []
    forces = []
    lattices = []
    po, fo = read_outcar_a_position_force(f, n)
    en = read_outcar_a_energy(f)
    try:
        while True:
            la = read_outcar_a_lattice(f)
            po, fo = read_outcar_a_position_force(f, n)
            en = read_outcar_a_energy(f)
            energies.append(en)
            positions.append(po)
            forces.append(fo)
            lattices.append(la)
    except StopIteration:
        return lattices, symbols, positions, forces, energies


@multimethod
def read_outcar_trajectory(path: str):
    with open(path) as f:
        return read_outcar_trajectory(f)
