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

from codes.lib.fc.fc_generic import fc_parallel


# IDTxl parameters
def parm_append_cmi(param, cmiEst):
    return {**param, **{'cmi_estimator'   : cmiEst}}


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


def write_phase_space_fig(data, dataName):
    plt.figure()
    plt.plot(data[0].flatten(), data[1].flatten(), '.')
    plt.savefig("data_" + dataName + ".png")
    plt.close()


def write_fc_lag_p_fig(rez, rezLabel):
    # Plot results
    fig, ax = plt.subplots(ncols=len(rez), squeeze=False)
    fig.suptitle(rezLabel)
    for iRez, rezMat in enumerate(rez):
        rezAbsRound = np.round(np.abs(rezMat), 2)
        ax[0, iRez].imshow(rezAbsRound, vmin=0)
        for (j, i), label in np.ndenumerate(rezAbsRound):
            ax[0, iRez].text(i, j, label, ha='center', va='center', color='r')

    plt.savefig(rezLabel + ".png")
    plt.close()


########################
# Generic parameters
########################
param = {
    'dim_order'       : 'prs',
    'max_lag_sources' : 0,
    'min_lag_sources' : 0
}

# Test Linear
nTime = 10
nTrial = 10
nCoreArr = np.arange(4, 5)

########################
# Sweep parameters
########################

algDict = {
    "Libraries"    : ["corr", "corr", "idtxl", "idtxl", "idtxl"],
    "Estimators"   : ["corr", "spr", "BivariateMI", "BivariateMI", "BivariateMI"],
    "CMI"          : [None, None, "OpenCLKraskovCMI", "JidtGaussianCMI", "JidtKraskovCMI"],
    "parallel_trg" : [False, False, True, True, True]
}


for iTestRepetition in range(20):

    timesDict = {}
    ptests = []
    for dataName in ["Linear", "Circular"]:
        data = generate_data(nTrial, nTime, False, dataName == "Linear", False)

        #write_phase_space_fig(data, dataName)

        for library, estimator, cmi, parTarget in zip(*algDict.values()):
            # Get label
            taskKey = (dataName, library, estimator, cmi)
            print("computing", taskKey)

            # Get settings
            paramThis = param if cmi is None else parm_append_cmi(param, cmi)

            # Compute performance metric
            timesDict[taskKey] = []

            for nCore in nCoreArr:
                tStart = time()
                rez = fc_parallel(data, library, estimator, paramThis, parTarget=parTarget, serial=False, nCore=nCore)
                timesDict[taskKey] += [time() - tStart]
                ptests += [taskKey + (nCore, rez[2][0,1])]

                #write_fc_lag_p_fig(rez, str(taskKey) )

            # TODO: Report significance of off-diagonal link


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