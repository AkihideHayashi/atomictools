import numpy as np
from atomictools.tools import read_matrix
from atomictools.unit.au import angstrom

class AXSF(object):
    def __init__(self):
        self.animsteps = 1
        self.periodicity = "MOLECULE"
        self.primvec = []
        self.convvec = []
        self.number = []
        self.coordinate = []
        self.force = []
    
    @staticmethod
    def read(file):
        if isinstance(file, str):
            with open(file) as f:
                return AXSF.read(f)
        axsf = AXSF()
        
        fp = (line.strip().split("#")[0] for line in file)
        fp = (line for line in fp if line)
        label = next(fp).split()
        while True:
            try:
                label = axsf.read_label(fp, label)
            except StopIteration:
                break
        return axsf
    
    def read_label(self, f, label):
        if label[0] in ("CRYSTAL", "SLAB", "POLYMER", "MOLECULE"):
            assert len(label) == 1
            self.periodicity = label[0]
            return next(f).split()
        elif label[0] == "ANIMSTEPS":
            self.animsteps = int(label[1])
            return next(f).split()
        elif label[0] == "ATOMS":
            return self.read_atoms(f)
        elif label[0] == "PRIMVEC":
            return self.read_primvec(f)
        elif label[0] == "CONVVEC":
            return self.read_convvec(f)
        elif label[0] == "PRIMCOORD":
            return self.read_primcoord(f)
        else:
            raise NotImplementedError(label)
            
    def read_atoms(self, fp):
        number = []
        coordinate = []
        force = []
        try:
            while True:
                line = next(fp).split()
                if line[0] == "ATOMS":
                    return line
                else:
                    n, c, f = read_atomic_line(line)
                    number.append(n)
                    coordinate.append(c)
                    force.append(f)
        finally:
            self.number.append(np.array(number))
            self.coordinate.append(np.array(coordinate))
            self.force.append(np.array(force))
    
    def read_primvec(self, f):
        self.primvec.append(read_matrix(f, 3).astype(np.float64) * angstrom)
        return next(f).split()
    
    def read_convvec(self, f):
        self.convvec.append(read_matrix(f, 3).astype(np.float64) * angstrom)
        return next(f).split()
    
    def read_primcoord(self, fp):
        n_atoms, _ = map(int, next(fp).split())
        number = []
        coordinate = []
        force = []
        for _ in range(n_atoms):
            n, c, f = read_atomic_line(next(fp).split())
            number.append(n)
            coordinate.append(c)
            force.append(f)
        self.number.append(np.array(number))
        self.coordinate.append(np.array(coordinate))
        self.force.append(np.array(force))
        return next(fp).split()
    
    
def read_atomic_line(line):
    if len(line) == 4:
        n, x, y, z = line
        dx, dy, dz = 0.0, 0.0, 0.0
    elif len(line) == 7:
        n, x, y, z, dx, dy, dz = line
    return int(n), np.array([x, y, z]).astype(np.float64) * angstrom, np.array([dx, dy, dz]).astype(np.float64) / angstrom