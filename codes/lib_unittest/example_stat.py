import numpy as np
import matplotlib.pyplot as plt
import codes.lib.stat.stat_lib as stat_lib

plt.ion()

############################
# Random bool permutation
############################
N_TEST1 = 1000
nTot = 100
freqArr = np.zeros(nTot)
nTrueArr = np.random.randint(0, 100, N_TEST1)

for nTrue in zip(nTrueArr):
    perm = stat_lib.rand_bool(nTrue, nTot).astype(int)
    assert np.sum(perm) == nTrue
    freqArr += perm

plt.figure()
plt.title("Testing array permutation for bias")
plt.xlabel("array element")
plt.ylabel("number of occurences")
plt.plot(freqArr, '.')
plt.show()

############################
# Discrete PDF and CDF
############################
N_VAL_TEST2 = 20
keys = np.random.randint(0, 1000, N_VAL_TEST2)
vals = np.random.uniform(0, 1, N_VAL_TEST2)
vals /= np.sum(vals)

pdf = dict(zip(keys, vals))
cdf = stat_lib.discrete_distr_to_cdf(pdf)

plt.figure()
plt.title("Discrete PDF and CDF")
plt.semilogy(pdf.keys(), pdf.values(), '.')
plt.semilogy(cdf.keys(), cdf.values())
plt.show()

############################
# Sampling distributions
############################
N_TEST3 = 1000
N_VAL_TEST3 = 20
keys = np.arange(N_VAL_TEST3)
vals = np.random.uniform(0,1,N_VAL_TEST3)
vals /= np.sum(vals)
pdf = dict(zip(keys,vals))
cdf = stat_lib.discrete_distr_to_cdf(pdf)

resample = stat_lib.discrete_resample(cdf, N_TEST3)
emp_pdf = stat_lib.discrete_empirical_pdf_from_sample(resample)

assert emp_pdf.keys() == pdf.keys()

plt.figure()
plt.title("Resamling random distribution using " + str(N_TEST3) + " points")
plt.bar(pdf.keys(), pdf.values(), alpha=0.5, label="true")
plt.bar(emp_pdf.keys(), emp_pdf.values(), alpha=0.5, label="resampled")
plt.legend()
plt.ioff()
plt.show()