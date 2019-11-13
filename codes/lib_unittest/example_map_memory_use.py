import sys
import time
import numpy as np
from codes.lib.aux_functions import memNowAsStr


def heavyFunc(i):
    x = np.random.normal(0,1, 10**8)
    print(i, memNowAsStr(), sys.getsizeof(x))
    return i


print(list(map(heavyFunc, np.arange(10))))