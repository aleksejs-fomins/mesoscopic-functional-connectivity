import numpy as np
import codes.lib.stat.resample_lib as resample_lib

# Evaluate in which percentile of the test data does the true value lay
# pvalL - probability of getting a random value at least as small as true
# pvalR - probability of getting a random value at least as large as true
# Note: The test cannot guarantee a pValue below 1/nSample, so in case of zero matches the pValue is upper-bounded
def percentile_twosided(fTrue, fTestArr):
    nSample = fTestArr.shape[0]
    pvalL = np.max([np.mean(fTestArr <= fTrue), 1/nSample])
    pvalR = np.max([np.mean(fTestArr >= fTrue), 1/nSample])
    return pvalL, pvalR

# Test if relative order of X vs Y matters.
# Tests if values of f(x,y) are significantly different if y is permuted wrt x
def paired_test(f, x, y, nSample):
    assert x.shape == y.shape
    fTrue = f(x, y)
    fTestArr = resample_lib.resample_dyad_individual(f, x, y, nSample=nSample, method="permutation")
    return percentile_twosided(fTrue, fTestArr)



# Tests whether X and Y came from the same distribution
# The values of X and Y are resampled from the shared pool
def difference_test(f, x, y, nSample):
    assert x.shape == y.shape
    fTrue = f(x, y)
    fTestArr = resample_lib.resample_dyad_union(f, x, y, nSample, method="resample")
    return percentile_twosided(fTrue, fTestArr)
