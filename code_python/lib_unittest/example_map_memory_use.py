import sys
import numpy as np
from lib.sys_lib import mem_now_as_str


def heavyFunc(i):
    x = np.random.normal(0,1, 10**8)
    print(i, mem_now_as_str(), sys.getsizeof(x))
    return i


print(list(map(heavyFunc, np.arange(10))))