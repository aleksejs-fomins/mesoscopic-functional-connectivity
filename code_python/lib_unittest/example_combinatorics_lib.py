import numpy as np

from lib.stat.comb_lib import log_comb, comb_fak

print("Computing rel. error of approximate number of combinations for small values")
for i in range(20):
    for j in range(i):
        combExact = comb_fak(i, j)
        combApprox = np.exp(log_comb(i, j))
        print(i,j, combApprox/combExact - 1)
