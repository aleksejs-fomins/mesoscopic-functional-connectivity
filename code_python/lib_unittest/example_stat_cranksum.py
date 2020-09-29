
import numpy as np
import matplotlib.pyplot as plt

from lib.stat.test_lib import difference_test

A = np.random.uniform(0.5,2,100)#np.abs(np.random.normal(0,1,100))
B = np.random.uniform(0,1,100)#np.abs(np.random.normal(0,1,100))

def CDF(x):
    return np.sort(x), np.linspace(0,1,len(x))

def permutation_test_distinct(f, x, y, nResample=2000):
    M, N = len(x), len(y)
    fTrue = f(x, y)
    xy = np.hstack([x, y])
    fTest = []
    for iTest in range(nResample):
        xyShuffle = xy[np.random.permutation(M + N)]
        xTest, yTest = xyShuffle[:M], xyShuffle[M:]
        fTest += [f(xTest, yTest)]

    plt.figure()
    plt.hist(fTest, bins='auto')
    plt.axvline(x=fTrue, color='r')




xa, ca = CDF(A)
xb, cb = CDF(B)

fig, ax = plt.subplots()
ax.plot(xa, ca)
ax.plot(xb, cb)

permutation_test_distinct(cranksum, A, B)
plt.show()

