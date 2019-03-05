# pylint: disable=E0102
import numpy as np
from numpy import pi
from numpy.linalg import det
from multimethod import multimethod
import io

from atomictools.tools import contract, expand, read_matrix, skip_until, read_aslongas, fmt, withopen
from atomictools.unit.au import kayser, eV, angstrom
from atomictools.lattice import move_to_lattice
from atomictools.atomic import atomic_number, mass, mass_da
from atomictools.freeenergy import gibbs_translation, helmholtz_rotation, helmholtz_vibration, tensor_of_inertia
from atomictools.freeenergy import energy_rotation, energy_translation, energy_vibration
from atomictools.freeenergy import entropy_rotation, entropy_translation, entropy_vibration

class Poscar(object):
    def __init__(self, title, unit, lattice, symbols, coordinates, selective_dynamics):
        self.title = title
        self.unit = unit
        self.lattice = lattice
        self.symbols = symbols
        self.coordinates = coordinates
        self.selective_dynamics = selective_dynamics

    @property
    def numbers(self):
        return atomic_number(self.symbols)

    @property
    def mass(self):
        return mass[self.numbers]

    @property
    def mass_da(self):
        return mass_da[self.numbers]

    @staticmethod
    def read(f):
        return Poscar(*read_poscar(f))

    def write(self, f, cartesian, mode=None):
        if mode is not None:
            write_poscar(
                f,
                self.title,
                self.unit,
                self.lattice,
                self.symbols,
                cartesian,
                self.coordinates,
                self.selective_dynamics,
                mode
                )
        else:
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

    def move_to_lattice(self):
        self.coordinates = move_to_lattice(self.coordinates, self.lattice)
    
    def extend(self, symbols, coordinates, selective_dynamics=None):
        self.coordinates = np.array(self.coordinates.tolist() + coordinates.tolist())
        self.symbols = np.array(self.symbols.tolist() + symbols.tolist())
        if self.selective_dynamics is not None:
            self.selective_dynamics = np.array(self.selective_dynamics.tolist() + selective_dynamics.tolist())

    def sort(self, key=lambda scd: atomic_number(scd[0]), **kwargs):
        """key should take scd = (symbol, coordinate, [selective_dynamics])
        """
        if self.selective_dynamics is None:
            self.symbols, self.coordinates = [np.array(list(x)) for x in zip(*sorted(zip(self.symbols, self.coordinates), key=key, **kwargs))]
        else:
            self.symbols, self.coordinates, self.selective_dynamics = [np.array(list(x)) for x in zip(*sorted(zip(self.symbols, self.coordinates, self.selective_dynamics), key=key, **kwargs))]

    def extend_cell(self, n):
        extend_cell(self, n)
        

def extend_cell(pos, n):
    for i, ni in enumerate(n):
        lattice = pos.lattice.copy()[i]
        coordinates = pos.coordinates.copy()
        symbols = pos.symbols.copy()
        selective = pos.selective_dynamics
        for j in range(1, ni):
            pos.extend(symbols, coordinates + lattice * j, selective)
        pos.lattice[i] *= ni 


class OutcarTrajectory(object):
    def __init__(self, symbols, lattices, positions, forces, energies):
        self.symbols = symbols
        self.lattices = lattices
        self.positions = positions
        self.forces = forces
        self.energies = energies
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            z = zip(self.lattices[item], self.positions[item], self.forces[item], self.energies[item])
            return [(self.symbols, *a) for a in z]
        else:
            return self.symbols, self.lattices[item], self.positions[item], self.forces[item], self.energies[item]
        
    def __iter__(self):
        return iter((self.symbols, l, p, f, e) for (l, p, f, e) in zip(self.lattices, self.positions, self.forces, self.energies))
    
    def __len__(self):
        return len(self.lattices)
    
    @staticmethod
    def read(path):
        return OutcarTrajectory(*read_outcar_trajectory(path))


class OutcarFrequency(object):
    def __init__(self, lattice, symbols, Uel, im, en, coordinates, frequencies, atomic, sigma):
        self.lattice = lattice
        self.symbols = symbols
        self.im = im
        self.Uel = Uel
        self.en = en
        self.coordinates = coordinates
        self.frequencies = frequencies
        self.numbers = atomic_number(self.symbols)
        self.mass = mass[self.numbers]
        self.I = tensor_of_inertia(self.coordinates[0], self.mass)
        self.total_mass = sum(self.mass)
        self.is_linear = abs(det(self.I)) < 1E-8
        self.vib_free = len(self.en) - 5 if self.is_linear else len(self.en) - 6
        self.atomic = atomic
        self.sigma = sigma

    @staticmethod
    def read(path, atomic, sigma):
        return OutcarFrequency(*read_outcar_frequency(path), atomic, sigma)

    def free_energy(self, beta, pressure=None):
        if self.atomic:
            G_tra = gibbs_translation(self.total_mass / (2 * pi), beta, pressure)
            A_rot = helmholtz_rotation(self.I * 2, beta, self.sigma)
            A_vib = helmholtz_vibration(self.en[:self.vib_free], beta)
            return self.Uel + G_tra + A_rot + A_vib
        else:
            A_vib = helmholtz_vibration(self.en, beta)
            return self.Uel + A_vib

    def internal_energy(self, beta, pressure=None):
        if self.atomic:
            U_tra = energy_translation(beta)
            U_rot = energy_rotation(self.I * 2, beta)
            U_vib = energy_vibration(self.en[:self.vib_free], beta)
            return self.Uel + U_tra + U_rot + U_vib
        else:
            U_vib = energy_vibration(self.en, beta)
            return self.Uel + U_vib

    def enthalpy(self, beta, pressure=None):
        if self.atomic:
            return self.internal_energy(beta, pressure) + 1 / beta
        else:
            return self.internal_energy(beta, pressure)
    
    def entropy(self, beta, pressure=None):
        if self.atomic:
            S_tra = entropy_translation(self.total_mass / (2 * pi), beta, pressure)
            S_rot = entropy_rotation(self.I * 2, beta, self.sigma)
            S_vib = entropy_vibration(self.en[:self.vib_free], beta)
            return S_tra + S_rot + S_vib
        else:
            S_vib = entropy_vibration(self.en, beta)
            return S_vib


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
        # selectived = np.full((len(symbols), 3), True)]
        selectived = None
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
def write_poscar(path: str, title, unit, lattice, symbols, cartesian, positions, selective, mode=None):
    with open(path, "w") as f:
        write_poscar(f, title, unit, lattice, symbols, cartesian, positions, selective)
        f.write("\n")
        if mode is not None:
            mod = fmt("{:<016.10}", mode / unit)
            for m in mod:
                f.write("  ".join(m) + "\n")
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
        return symbols, lattices, positions, forces, energies


@multimethod
def read_outcar_trajectory(path: str):
    with open(path) as f:
        return read_outcar_trajectory(f)


def read_outcar_a_frequency(f: io.TextIOWrapper, n):
    head = next(f).split()
    if head[1] == "f":
        body = head[3:]
        im = False
    elif head[1] == "f/i=":
        body = head[2:]
        im = True
    else:
        raise NotImplementedError()
    energy = float(body[4]) * kayser
    next(f)
    pos_freq = read_matrix(f, n).astype(np.float64)
    next(f)
    return im, energy, pos_freq


@withopen
def read_outcar_frequency(f: io.TextIOWrapper):
    skip_until(f, "INCAR:")
    s = []
    while True:
        line = next(f)
        if "POTCAR:" in line:
            s.append(line.split()[2])
        if "ions per type =" in line:
            n = [int(w) for w in line.split("=")[1].split()]
            break
    symbols = expand(n, s)
    n = len(symbols)
    lattice = read_outcar_a_lattice(f)
    en0 = float(skip_until(f, "energy  without entropy").split()[-1]) * eV
    degree_of_freedom = int(skip_until(f, "Degree of freedom:").split()[-1])
    skip_until(f, "Eigenvectors and eigenvalues of the dynamical matrix")
    for _ in range(3):
        next(f)
    im, en, pf = map(np.array, zip(*[read_outcar_a_frequency(f, n) for _ in range(degree_of_freedom)]))
    return lattice, symbols, en0, im, en, pf[:, :, :3] * angstrom, pf[:, :, 3:]


class Doscar(object):
    s = 1
    py = 2
    pz = 3
    px = 4
    dxy = 5
    dyz = 6
    dz2r2 = 7
    dxz = 8
    dx2y2 = 9
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
    # pylint: disable=W0612
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
        PDOS = np.array(PDOS)
    else:
        PDOS = None
    return Emax, Emin, NEDOS, Efermi, DOS, PDOS


@multimethod
def read_doscar(path: str):
    with open(path) as f:
        return read_doscar(f)


@withopen
def read_core_state_eigenenergies(f):
    line = skip_until(f, "the core state eigenenergies are")
    i = 1
    ret = []
    while True:
        line = next(f).strip()
        if line:
            splt = line.split()
            if splt[0] == "{}-".format(i):
                i += 1
                ret.append(dict())
                splt = splt[1:]
            for j in range(len(splt) // 2):
                ret[-1][splt[j * 2]] = float(splt[j * 2 + 1])
        else:
            break
    return ret