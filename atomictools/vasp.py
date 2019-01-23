import numpy as np
from multimethod import multimethod
import io

from atomictools.tools import contract, expand, read_matrix, skip_until, read_aslongas, fmt


class Poscar(object):
    def __init__(self, title, unit, lattice, symbols, coordinates, selective_dynamics):
        self.title = title
        self.unit = unit
        self.lattice = lattice
        self.symbols = symbols
        self.coordinates = coordinates
        self.selective_dynamics = selective_dynamics

    @staticmethod
    def read(f):
        return Poscar(*read_poscar(f))

    def write(self, f, cartesian):
        write_poscar(
            f,
            self.title,
            self.unit,
            self.lattice,
            self.symbols,
            cartesian,
            self.coordinates,
            self.selective_dynamics
            )


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
    """system, cell, symbols, coordinate, selective"""
    title = next(f).strip()
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
    return title, unit, cell * unit, symbols, coordinate * unit, selectived


@multimethod
def read_poscar(path: str):
    with open(path) as f:
        return read_poscar(f)
    return


@multimethod
def write_poscar(f: io.TextIOWrapper, title, unit, lattice, symbols, cartesian, positions, selective):
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
    if cartesian:
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
def write_poscar(path: str, title, unit, lattice, symbols, cartesian, positions, selective):
    with open(path, "w") as f:
        write_poscar(f, title, unit, lattice, symbols, cartesian, positions, selective)
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


class Doscar(object):
    def __init__(self, Emax, Emin, NEDOS, Efermi, DOS, PDOS):
        self.e_max = Emax
        self.e_min = Emin
        self.nedos = NEDOS
        self.e_fermi = Efermi
        self.dos = DOS
        self.pdos = PDOS
    
    @staticmethod
    def read(path):
        return Doscar(*read_doscar(path))
    

@multimethod
def read_doscar(f: io.TextIOWrapper):
    (number_of_ions_including_empty_spheres,
     number_of_ions,
     including_pdos,
     NCDIJ) = map(int, next(f).split())
    (volume_of_unit_cell,
     la, lb, lc,
     POTIM) = map(float, next(f).split())
    TEBEG = float(next(f))
    next(f)
    SYSTEM = next(f)
    Emax, Emin, FEDOS, Efermi, _ = map(float, next(f).split())
    NEDOS = int(FEDOS)
    DOS = read_matrix(f, NEDOS).astype(np.float64)
    if including_pdos:
        PDOS = []
        for _ in range(number_of_ions):
            next(f)
            PDOS.append(read_matrix(f, NEDOS).astype(np.float64))
    else:
        PDOS = None
    return Emax, Emin, NEDOS, Efermi, DOS, PDOS


@multimethod
def read_doscar(path: str):
    with open(path) as f:
        return read_doscar(f)