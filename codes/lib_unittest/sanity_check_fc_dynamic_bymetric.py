import os, sys
from time import time
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


# Plot phase-space of different shifts of both variables wrt each other. Produces one plot [nSWEEP x 5]
class phaseSpaceFigures():
    def __init__(self, nSweep):
        self.fig1, self.axData = plt.subplots(nrows=nSweep, ncols=5, tight_layout=True)
        self.fig1.suptitle("Phase space plots")
        self.axData[0, 0].set_title("Self-prediction, data1")
        self.axData[0, 1].set_title("Self-prediction, data2")
        self.axData[0, 2].set_title("cross-prediction, lag=0")
        self.axData[0, 3].set_title("cross-prediction 1->2, lag=1")
        self.axData[0, 4].set_title("cross-prediction 2->1, lag=1")

    def addrow(self, data, dataLabel):
        xFlatPre = data[0, :, :-1].flatten()
        yFlatPre = data[1, :, :-1].flatten()
        xFlatPost = data[0, :, 1:].flatten()
        yFlatPost = data[1, :, 1:].flatten()

        self.axData[iSweep, 0].set_ylabel(dataLabel)
        self.axData[iSweep, 0].plot(xFlatPre, xFlatPost, '.')
        self.axData[iSweep, 1].plot(yFlatPre, yFlatPost, '.')
        self.axData[iSweep, 2].plot(xFlatPre, yFlatPre, '.')
        self.axData[iSweep, 3].plot(xFlatPre, yFlatPost, '.')
        self.axData[iSweep, 4].plot(yFlatPre, xFlatPost, '.')


# Plot FC, LAG, and P-VAL matrices for each SWEEP and ALG combination. Produces 3 plots
class metricFigures():
    def __init__(self, figKeys, nSweep, algKeys):
        nAlg = len(algKeys)

        self.figs = []
        self.axes = []
        for figKey in figKeys:
            fig, ax = plt.subplots(nrows=nSweep, ncols=nAlg, tight_layout=True)
            fig.suptitle("Functional Connectivity " + figKey)

            for iAlg, algKey in enumerate(algKeys):
                ax[0, iAlg].set_title(algKey)

            self.figs += [fig]
            self.axes += [ax]


    def set_row_label(self, iSweep, dataLabel):
        for ax in self.axes:
            ax[iSweep, 0].set_ylabel(dataLabel)


    def addcell(self, rezLst, iSweep, iAlg):
        for iRez, (rez, fig, ax) in enumerate(zip(rezLst, self.figs, self.axes)):
            rezAbs = np.round(np.abs(rez), 3)
            if iRez == 0:
                ax[iSweep, iAlg].imshow(rezAbs, vmin=0)
            else:
                ax[iSweep, iAlg].imshow(rezAbs, vmin=0, vmax=1)

            for (j, i), label in np.ndenumerate(rezAbs):
                ax[iSweep, iAlg].text(i, j, label, ha='center', va='center', color='r')


########################
# Generic parameters
########################
param = {
    'dim_order'       : 'prs',
    'max_lag_sources' : 1,
    'min_lag_sources' : 1
}

# Test Linear
nCore = 4
nTime = 1000
nTrial = 5

########################
# Sweep parameters
########################

algDict = {
    "Libraries"    : ["corr", "corr", "idtxl", "idtxl", "idtxl", "idtxl"],
    "Estimators"   : ["corr", "spr", "BivariateMI", "BivariateMI", "BivariateTE", "BivariateTE"],
    "CMI"          : [None, None, "JidtGaussianCMI", "JidtKraskovCMI", "JidtGaussianCMI", "JidtKraskovCMI"],
    "parallel_trg" : [False, False, True, True, True, True]
}

# Define algorithm name as concatenation of all its properties that are strings
algNames = ["_".join([val for val in algVals if type(val) == str]) for algVals in zip(*algDict.values())]

varSweep = [(p,d,s) for p in ["incr", "random"] for d in ["linear", "circular"] for s in ["matching", "shifted"]]
nSweep = len(varSweep)

########################
# Figures
########################

psFigs = phaseSpaceFigures(nSweep)
metricFigs = metricFigures(["values", "lags", "p-values"], nSweep, algNames)


########################
# Sweep loop
########################

performanceTimes = []
testPValues = []

for iSweep, var in enumerate(varSweep):
    predictName, dataName, shiftName = var
    data = generate_data(nTrial, nTime, predictName=="incr", dataName == "linear", shiftName=="shifted")
    dataLabel = predictName + "_" + dataName + "_" + shiftName
    psFigs.addrow(data, dataLabel)
    metricFigs.set_row_label(iSweep, dataLabel)

    for iAlg, (algName, library, estimator, cmi, parTarget) in enumerate(zip(algNames, *algDict.values())):
        # Get label
        resultLabel = dataLabel + "_" + algName
        print("computing", resultLabel)

        # Get settings
        paramThis = param if cmi is None else parm_append_cmi(param, cmi)

        # Compute metric
        tStart = time()
        rez = fc_parallel(data, library, estimator, paramThis, parTarget=parTarget, serial=False, nCore=nCore)
        performanceTimes += [(var, algName, time() - tStart)]

        # Plot results
        metricFigs.addcell(rez, iSweep, iAlg)

        # Report significance of off-diagonal link
        testPValues += [(resultLabel, rez[2][0, 1])]


print("Off diagonal p-values")
print(np.array(testPValues))

print("Timing in seconds")
print(np.array(performanceTimes))

plt.show()
