from pathlib import Path
import subprocess as sub
import tempfile
import os
from multimethod import multimethod
from atomictools.xyz import write_xyz

def levelup(s):
    n = 4
    return " " * n + s.replace("\n", "\n" + " " * n)

def conversion(s):
    if s is True:
        return "Yes"
    elif s is False:
        return "No"
    elif isinstance(s, dict):
        return "{{\n{}\n}}".format(levelup("\n".join("{} = {}".format(key, conversion(val)) for key, val in s.items())))
    elif isinstance(s, str):
        return "\"{}\"".format(s)
    elif isinstance(s, Path):
        return s.as_posix()
        # return s.name
    else:
        return s

class DFTBP(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = conversion(val)
    
    @property
    def contents(self):
        return "\n".join("{} = {}".format(key, val) for key, val in self.__dict__.items())
       
    def __repr__(self):
        return "{}{{\n{}\n}}".format(self.__class__.__name__, levelup(self.contents))

class GenFormat(DFTBP):
    def __init__(self, path):
        self.path = path
    
    @property
    def contents(self):
        return "<<< \"{}\"".format(self.path.as_posix())
    
class ConjugateGradient(DFTBP):
    pass

class SecondDerivatives(DFTBP):
    pass
    
class DFTB(DFTBP):
    pass
    
class Type2FileNames(DFTBP):
    pass

class Options(DFTBP):
    pass

class ParserOptions(DFTBP):
    pass

class Analysis(DFTBP):
    pass

class DFTBPlus(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __repr__(self):
        ret = []
        for key, val in self.kwargs.items():
            ret.append("{} = {}".format(key, val))
        for val in self.args:
            ret.append("{}".format(val))
        return "\n".join(ret)


def write_gen(path, symbols, coordinates, lattice=None, fractional=False):
    tmpn1, tmps1 = tempfile.mkstemp()
    command = ["xyz2gen", tmps1]
    with os.fdopen(tmpn1, "w") as tmpf:
        write_xyz(tmpf, "tmp", symbols, coordinates)
    if lattice is not None:
        tmpn2, tmps2 = tempfile.mkstemp()
        with os.fdopen(tmpn2, "w") as tmpf:
            for l in lattice:
                tmpf.write("{}\n".format("  ".join(map(str, l))))
        command.extend(["-l", tmps2])
    command.extend(["-o", path])
    if fractional:
        command.append("-f")

    sub.run(" ".join(command), shell=True)
    
    os.remove(tmps1)
    if lattice is not None:
        os.remove(tmps2)

        
def read_a_results_tag(f: io.TextIOWrapper):
    name, tds = next(f).split()
    _, t, d, s = tds.split(":")
    dim = int(d)
    shape = tuple(map(int, s.split(",")))
    if t == "real":
        dtype = np.float64
    else:
        raise RuntimeError()
    vec = read_matrix(f, int(np.ceil(np.prod(shape) / 3))).astype(np.float64)
    return name, vec.reshape(shape)


@multimethod
def read_results_tag(f: io.TextIOWrapper):
    ret = dict()
    while True:
        try:
            name, x = read_a_results_tag(f)
            ret[name] = x
        except StopIteration:
            break
    return ret
       
    
@multimethod
def read_results_tag(path: str):
    with open(path) as f:
        return read_results_tag(f)