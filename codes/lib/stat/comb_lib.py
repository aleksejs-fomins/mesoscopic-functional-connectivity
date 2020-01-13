'''
Combinatorics library
'''

from math import lgamma
from scipy.misc import comb

def comb_fak(N, k):
    return comb(N, k, exact=True)

# Compute the logarithm of number of combinations. Approximate, but works for large N
def log_comb(N, k):
    return lgamma(N+1) - lgamma(N-k+1) - lgamma(k+1)