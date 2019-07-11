import numpy as np
from multimethod import multimethod

def split_next(f):
    while True:
        line = next(f).strip().split("#")[0].split(None, 1)
        if line:
            return line
        else:
            continue
            
def read_cp2k_section(section, f):
    ret = []
    while True:
        splitted = split_next(f)
        if splitted[0][0] == '&':
            if splitted[0][1:4] == 'END':
                if len(splitted) > 1:
                    assert splitted[1].strip() == section[0][1:].strip()
                return Section(section, ret)
            else:
                ret.append(read_cp2k_section(splitted, f))
        else:
            ret.append(tuple(splitted))
            
class CP2K(object):
    def __init__(self, sections):
        self.sections = sections

    def __repr__(self):
        return "\n".join(s.dump() for s in self.sections)

    def __getitem__(self, key):
        for c in self.sections:
            if c.section[0] == key:
                return c
        raise KeyError()
        
    @staticmethod
    def read(f):
        if isinstance(f, str):
            with open(f) as fp:
                return CP2K.read(fp)
        ret = []
        while True:
            try:
                section = split_next(f)
            except StopIteration:
                break
            if section[0][0] == '&':
                ret.append(read_cp2k_section(section, f))
            else:
                break
        return CP2K(ret)
    
    def write(self, f):
        if isinstance(f, str):
            with open(f, 'w') as fp:
                self.write(fp)
        else:
            f.write(repr(self))

class Section(object):
    def __init__(self, section, contents):
        self.section = section
        self.contents = contents
    
    def dump(self, join=True):
        section = " ".join(self.section)
        contents = []
        for c in self.contents:
            if isinstance(c, Section):
                contents.extend(c.dump(False))
            else:
                contents.append(" ".join(c))
        ret = [section] + ["  " + c for c in contents] + [f"&END {self.section[0][1:]}"]
        if join:
            return "\n".join(ret)
        else:
            return ret
    def __repr__(self):
        return self.dump()

    def __getitem__(self, key):
        for c in self.contents:
            if isinstance(c, Section):
                if c.section[0] == key:
                    return c
            else:
                if c[0] == key:
                    return c[1:]
        raise KeyError(key)
