from time import time
import numpy as np
import matplotlib.pyplot as plt

from codes.lib.fc.corr_lib import crossCorr
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPU
# Datasets

# IDTxl parameters
def getIDTxlSettings(est, method):
    return {
        'dim_order'       : 'rsp',
        'method'          : method,
        'cmi_estimator'   : est,
        'max_lag_sources' : 1,
        'min_lag_sources' : 1
    }

algFuncDict = {
    "Corr"          : lambda data: crossCorr(data.transpose((0, 2, 1)), 1, 1, est='corr'),
    "Spr"           : lambda data: crossCorr(data.transpose((0, 2, 1)), 1, 1, est='spr'),
    "BMI_Gaussian"  : lambda data: idtxlParallelCPU(data.transpose((1, 2, 0)), getIDTxlSettings('JidtGaussianCMI', 'BivariateMI')),
    "BMI_Kraskov"   : lambda data: idtxlParallelCPU(data.transpose((1, 2, 0)), getIDTxlSettings('JidtKraskovCMI',  'BivariateMI')),
    "BTE_Gaussian"  : lambda data: idtxlParallelCPU(data.transpose((1, 2, 0)), getIDTxlSettings('JidtGaussianCMI', 'BivariateTE')),
    "BTE_Kraskov"   : lambda data: idtxlParallelCPU(data.transpose((1, 2, 0)), getIDTxlSettings('JidtKraskovCMI',  'BivariateTE'))
}


def generate_data(nTrial, nData, isSelfPredictive, isLinear, isShifted):
    nDataZealous = nData + 1
    dataShape = (nTrial, nDataZealous)

    if isSelfPredictive:
        src = np.outer(np.ones(nTrial), np.linspace(0, nDataZealous, nDataZealous) / nData)
    else:
        src = np.random.uniform(0, 1, dataShape)

    if isLinear:
        x = src
        y = 1 - src
    else:
        x = np.cos(2 * np.pi * src)
        y = np.sin(2 * np.pi * src)

    x += 0.2 * np.random.normal(0, 1, dataShape)
    y += 0.2 * np.random.normal(0, 1, dataShape)

    if isShifted:
        return np.array([x[:, 1:], y[:, :-1]])
    else:
        return np.array([x[:, :-1], y[:, :-1]])


# Test Linear
nTime = 1000
nTrial = 5
nAlg = len(algFuncDict)

performanceTimes = []
testPValues = []



fig1, axData = plt.subplots(nrows=8, ncols=5, tight_layout=True)
fig1.suptitle("Phase space plots")
axData[0, 0].set_title("Self-prediction, data1")
axData[0, 1].set_title("Self-prediction, data2")
axData[0, 2].set_title("cross-prediction, lag=0")
axData[0, 3].set_title("cross-prediction 1->2, lag=1")
axData[0, 4].set_title("cross-prediction 2->1, lag=1")

metricFigures = []
for iRez, rezName in enumerate(["values", "lags", "p-values"]):
    metricFigures += [plt.subplots(nrows=8, ncols=nAlg, tight_layout=True)]
    metricFigures[-1][0].suptitle("Functional Connectivity " + rezName)

    for iAlg, algName in enumerate(algFuncDict.keys()):
        metricFigures[-1][1][0, iAlg].set_title(algName)


varSweep = [(p,d,s) for p in ["incr", "random"] for d in ["linear", "circular"] for s in ["matching", "shifted"]]

for iSweep, var in enumerate(varSweep):
    predictName, dataName, shiftName = var
    data = generate_data(nTrial, nTime, predictName=="incr", dataName == "linear", shiftName=="shifted")
    dataLabel = predictName + "_" + dataName + "_" + shiftName

    xFlatPre = data[0, :, :-1].flatten()
    yFlatPre = data[1, :, :-1].flatten()
    xFlatPost = data[0, :, 1:].flatten()
    yFlatPost = data[1, :, 1:].flatten()

    axData[iSweep, 0].set_ylabel(dataLabel)
    axData[iSweep, 0].plot(xFlatPre, xFlatPost, '.')
    axData[iSweep, 1].plot(yFlatPre, yFlatPost, '.')
    axData[iSweep, 2].plot(xFlatPre, yFlatPre, '.')
    axData[iSweep, 3].plot(xFlatPre, yFlatPost, '.')
    axData[iSweep, 4].plot(yFlatPre, xFlatPost, '.')

    for metricFigure in metricFigures:
        metricFigure[1][iSweep, 0].set_ylabel(dataLabel)


    for iAlg, (algName, algFunc) in enumerate(algFuncDict.items()):
        resultLabel = dataLabel + "_" + algName
        print("computing", resultLabel)

        # Compute metric
        tStart = time()
        rez = algFunc(data)
        performanceTimes += [(var, algName, time() - tStart)]

        # Plot results
        for iRez, metricFigure in enumerate(metricFigures):
            rezAbs = np.round(np.abs(rez[iRez]), 3)
            fig, ax = metricFigure
            if iRez == 0:
                ax[iSweep, iAlg].imshow(rezAbs, vmin=0)
            else:
                ax[iSweep, iAlg].imshow(rezAbs, vmin=0, vmax=1)

            for (j, i), label in np.ndenumerate(rezAbs):
                ax[iSweep, iAlg].text(i, j, label, ha='center', va='center', color='r')

        # Report significance of off-diagonal link
        testPValues += [(resultLabel, rez[2][0, 1])]


print("Off diagonal p-values")
print(np.array(testPValues))

print("Timing in seconds")
print(np.array(performanceTimes))

plt.show()