import re

def parse_next(f):
    while True:
        ret = re.split("[!#]", next(f))[0].strip().split(None, 1)
        if len(ret) == 0:
            continue
        elif len(ret) == 1:
            ret.append("")
            return ret
        elif len(ret) == 2:
            return ret
        else:
            assert False
            
def read_cp2k_section(sub_key, sub_val, f):
    ret = []
    while True:
        key, val = parse_next(f)
        if key[0] == '&':
            if key[:4] == '&END':
                assert not val or val.split()[0] == sub_key, f"{sub_key}: {val.split()[0]}"
                return Subsection(sub_key, sub_val, ret)
            else:
                ret.append(read_cp2k_section(key[1:], val, f))
        else:
            ret.append(Keyword(key, val))

class CP2K(list):
    def __init__(self, children):
        super().__init__(children)
    
    def search(self, key):
        for c in self:
            if c.key.lower() == key.lower():
                return c
        return None
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        get = self.search(key)
        if get:
            return get
        else:
            raise KeyError(key)
            
    def __getattr__(self, attr):
        return self[attr]
    
    def dump(self):
        ret = []
        for c in self:
            ret.extend([f"{d}" for d in c.dump()])
        return ret
    
    @staticmethod
    def read(f):
        ret = []
        while True:
            try:
                key, val = parse_next(f)
                ret.append(read_cp2k_section(key[1:], val, f))
            except StopIteration:
                break
        return CP2K(ret)
    
    def __repr__(self):
        return '\n'.join(self.dump())
            
class Keyword(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        
    def dump(self):
        return [f"{self.key} {self.val}"]
    
    def __repr__(self):
        return f"{self.key} {self.val}"
            
class Subsection(CP2K):
    def __init__(self, key, val, children):
        super().__init__(children)
        self.key = key
        self.val = val
        
    def dump(self):
        ret = []
        ret.append(f"&{self.key} {self.val}")
        for c in self:
            ret.extend([f"  {d}" for d in c.dump()])
        ret.append(f"&END {self.key}")
        return ret
