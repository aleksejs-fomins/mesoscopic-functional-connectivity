import os, sys
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.fc.corr_lib import corr3D
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPU

def generate_data(nTrial, nData, isSelfPredictive, isLinear, isShifted):
    dataShape = (nTrial, nData)

    if isSelfPredictive:
        src = np.outer(np.ones(nTrial), np.linspace(0, 1, nData))
    else:
        src = np.random.normal(0, 1, dataShape)

    if isLinear:
        x = src
        y = 1 - 2 * src
    else:
        x = np.cos(2 * np.pi * src)
        y = np.sin(2 * np.pi * src)

    x += 0.2 * np.random.normal(0, 1, dataShape)
    y += 0.2 * np.random.normal(0, 1, dataShape)

    if isShifted:
        y = np.hstack(([y[-1]], y[:-1]))

    return np.array([x, y])


# IDTxl parameters
def getIDTxlSettings(est):
    return {
        'dim_order'       : 'rsp',
        'method'          : 'BivariateMI',
        'cmi_estimator'   : est,
        'max_lag_sources' : 0,
        'min_lag_sources' : 0
    }


algFuncDict = {
    "Corr" : lambda data, est, ncore: corr3D(data, est=est),
    "BMI"  : lambda data, est, ncore: idtxlParallelCPU(data.transpose((1, 2, 0)), getIDTxlSettings(est), NCore=ncore)
}


algEstLst = [
    # ("Corr", "corr"),
    # ("Corr", "spr"),
    # ("BMI", "OpenCLKraskovCMI"),
    ("BMI", "JidtGaussianCMI"),
    #("BMI", "JidtKraskovCMI"),
]

pvalFuncDict = {
    "Corr"  : lambda rez: rez[1][0, 1],
    "BMI"   : lambda rez: rez[2][0, 1]
}

# Test Linear
nTime = 10
nTrial = 10
nCoreArr = np.arange(4, 5)


for iStuff in range(20):

    timesDict = {}
    ptests = []
    for dataName in ["Linear", "Circular"]:
        data = generate_data(nTrial, nTime, False, dataName == "Linear", False)

        # plt.figure()
        # plt.plot(data[0].flatten(), data[1].flatten(), '.')
        # plt.savefig("data_" + dataName + ".png")
        # plt.close()

        for method, estimator in algEstLst:
            taskKey = (dataName, method, estimator)
            print("computing", taskKey)

            # Compute metric
            timesDict[taskKey] = []

            for nCore in nCoreArr:
                tStart = time()
                rez = algFuncDict[method](data, estimator, nCore)
                timesDict[taskKey] += [time() - tStart]
                ptests += [taskKey + (nCore, pvalFuncDict[method](rez))]

            # # Plot results
            # fig, ax = plt.subplots(ncols=len(rez), squeeze=False)
            # fig.suptitle(resultLabel)
            # for iRez, rezMat in enumerate(rez):
            #     rezAbsRound = np.round(np.abs(rezMat), 2)
            #     ax[0, iRez].imshow(rezAbsRound, vmin=0)
            #     for (j, i), label in np.ndenumerate(rezAbsRound):
            #         ax[0, iRez].text(i, j, label, ha='center', va='center', color='r')
            #
            # plt.savefig(resultLabel + ".png")
            # plt.close()

            # Report significance of off-diagonal link


            print("Sleeping...")
            sleep(1)


print("Off diagonal p-values")
print(np.array(ptests))

plt.figure()
plt.title("Timing in seconds")
for k,v in timesDict.items():
    plt.plot(nCoreArr, v, label=str(k))
plt.legend()
plt.show()