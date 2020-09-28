import numpy as np
import matplotlib.pyplot as plt
from codes.lib.stat.resample_lib import resample_monad


def _calc_ar_mle(dataArr):
    x = dataArr[:-1]
    y = dataArr[1:]
    Cxx = np.sum(x*x)
    Cyx = np.sum(x*y)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Nx = np.prod(x.shape)

    return np.linalg.solve(
        np.array([[Cxx, Sx],[Sx, Nx]]),
        np.array([Cyx, Sy])
    )


# Data shape [nTrial, nTime]
def test_ar_decay_const(data, nSample=1000, alphaThr=0.9):
    alphaArr, betaArr = np.array(resample_monad(_calc_ar_mle, data, nSample, method="bootstrap")).T
    print("For nSample =", nSample, "P[a >=", alphaArr, "] =", np.sum(alphaArr > alphaThr))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].hist(alphaArr, bins='auto')
    ax[1].hist(betaArr, bins='auto')
    plt.show()
