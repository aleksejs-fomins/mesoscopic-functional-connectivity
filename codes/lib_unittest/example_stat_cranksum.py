
import numpy as np
import matplotlib.pyplot as plt

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

def cranksum(x, y):
    M, N = len(x), len(y)
    jointData = np.hstack([x, y])
    jointSign = np.array([1.0]*M + [-1.0]*N)
    jointNorm = np.array([1.0/M]*M + [1.0/N]*N)
    idxOrder = np.argsort(jointData)
    ranks = np.arange(1, M+N+1).astype(float)**2
    normRanks = ranks * jointNorm[idxOrder]
    signedRanks = normRanks * jointSign[idxOrder]
    signedRankSum = np.sum(signedRanks)

    # Normalize Ranksum by its maximum, which depends on number of datapoints
    if signedRankSum > 0:
        signedRankSum /= np.sum(normRanks[-M:]) - np.sum(normRanks[:N])
    else:
        signedRankSum /= np.sum(normRanks[-N:]) - np.sum(normRanks[:M])

    return signedRankSum


xa, ca = CDF(A)
xb, cb = CDF(B)

fig, ax = plt.subplots()
ax.plot(xa, ca)
ax.plot(xb, cb)

permutation_test_distinct(cranksum, A, B)
plt.show()

