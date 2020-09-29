import numpy as np
import matplotlib.pyplot as plt

from lib.info_metrics.corr_lib import corr_2D, cross_corr_3D
from lib.stat import graph_lib


def plot_mean_variance_activity_by_learning(dataDB):
    nPlot = len(dataDB.mice)
    fig1, ax1 = plt.subplots(ncols=nPlot, figsize=(nPlot * 4, 4))
    fig2, ax2 = plt.subplots(ncols=nPlot, figsize=(nPlot * 4, 4))
    fig1.suptitle("Maximal Activity during entire day, by channel")
    fig2.suptitle("Mean Activity during entire day, by channel")

    for iPlot, mousename in enumerate(dataDB.mice):
        mouseData = dataDB.get_rows('neuro', {'mousename': mousename})
        if mouseData.shape[0] > 0:
            dataIdxs = list(mouseData["date"].index)

            channelMax = np.zeros((len(dataIdxs), 12))
            channelMean = np.zeros((len(dataIdxs), 12))
            for i, dataIdx in enumerate(dataIdxs):
                dataThis = dataDB.dataNeuronal[dataIdx]
                channelMax[i] = np.max(dataThis, axis=(0, 1))
                channelMean[i] = np.mean(dataThis, axis=(0, 1))

            ax1[iPlot].set_title(mousename)
            ax2[iPlot].set_title(mousename)
            ax1[iPlot].plot(channelMax)
            ax2[iPlot].plot(channelMean)
    plt.show()


def plot_correlation_connectivity_metrics(dataDB):
    testNames = [
        "Mean Synchronization",
        "STD Synchronization",
        "Maximal Offdiag Value",
        "Mean weighted in-degree",
        "STD weighted in-degree",
        "Mean weighted out-degree",
        "STD weighted out-degree",
        "Average normalized total CC",
        "Average non-normalized total CC",
        "Average normalized in CC",
        "Average non-normalized in CC",
        "Average normalized out CC",
        "Average non-normalized out CC"]
    nTest = len(testNames)

    fig, ax = plt.subplots(ncols=nTest, figsize=(nTest * 4, 4))

    for mousename in dataDB.mice:
        mouseData = dataDB.get_rows('neuro', {'mousename' : mousename})
        if mouseData.shape[0] > 0:
            dataIdxs = list(mouseData["date"].index)

            testResults = np.zeros((len(dataIdxs), nTest))
            for i, dataIdx in enumerate(dataIdxs):
                # Compute cross-correlation absolute value
                dataThis = dataDB.dataNeuronal[dataIdx].transpose(2, 1, 0)  # channel x time x trial for cross-corr
                cAbs = np.abs(cross_corr_3D(dataThis, 0, 0)[0])

                # Compute connectivity metrics
                testResults[i] = np.array([
                    *graph_lib.diagonal_dominance(cAbs),
                    np.max(graph_lib.offdiag(cAbs)),
                    np.mean(graph_lib.degree_in(cAbs)),
                    np.std(graph_lib.degree_in(cAbs)),
                    np.mean(graph_lib.degree_out(cAbs)),
                    np.std(graph_lib.degree_out(cAbs)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='tot', normDegree=True)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='tot', normDegree=False)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='in', normDegree=True)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='in', normDegree=False)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='out', normDegree=True)),
                    np.mean(graph_lib.clustering_coefficient(cAbs, kind='out', normDegree=False))
                ])

            for iTest in range(nTest):
                ax[iTest].plot(mouseData['deltaDaysCentered'], testResults[:, iTest], label=mousename)

    for iTest in range(nTest):
        ax[iTest].set_title(testNames[iTest])
        ax[iTest].set_xlabel("Days from start")
        # ax[iTest].legend()
    plt.show()


def plot_correlation_mean_cc_bytime(dataDB):
    trialKeys = ['iGO', 'iNOGO']
    nTrialKeys = len(trialKeys)

    for mousename in dataDB.mice[:1]:
        print(mousename)

        mouseData = dataDB.get_rows('neuro', {'mousename': mousename})
        if mouseData.shape[0] > 0:
            fig, ax = plt.subplots(ncols=nTrialKeys, figsize=(5 * nTrialKeys, 5))
            fig.suptitle(mousename)

            for iKey, trialKey in enumerate(trialKeys):
                for rowIdx, row in mouseData.iterrows():

                    dataLabel = row['mousekey']
                    trialIdx  = dataDB.dataTrials[rowIdx][trialKey]
                    dataThis  = dataDB.dataNeuronal[rowIdx][trialIdx - 1]

                    print("--", dataLabel, trialKey, len(trialIdx))

                    nTrial, nTime, nChannel = dataThis.shape

                    ccNoNorm = np.zeros(nTime)
                    for iTime in range(nTime):
                        corrAbs = np.abs(corr_2D(dataThis[:, iTime, :].T))
                        ccNoNorm[iTime] = np.mean(graph_lib.clustering_coefficient(corrAbs, normDegree=False))

                    ax[iKey].plot(ccNoNorm, label=dataLabel)

                ax[iKey].set_title(trialKey)
                ax[iKey].legend()
    plt.show()


def plot_performance_by_days(dataDB, outname=None, show=True):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    for mousename in dataDB.mice:
        mouseData = dataDB.get_rows('neuro', {'mousename' : mousename})
        if mouseData.shape[0] > 0:
            #dataIdxs = np.array(mouseData["date"].index)
            dataIdxs = np.array(mouseData.index)
            perf = dataDB.dataPerformance[dataIdxs]

            # ax[0].plot(dataDB.deltaDays[dataIdxs], perf, label=mousename)
            # ax[1].plot(dataDB.deltaDaysCentered[dataIdxs], perf, label=mousename)
            ax[0].plot(mouseData["deltaDays"], perf, label=mousename)
            ax[1].plot(mouseData["deltaDaysCentered"], perf, label=mousename)

    ax[0].set_title("Performance from start")
    ax[1].set_title("Performance centered at becoming expert")
    ax[0].set_xlabel("Days from start")
    ax[1].set_xlabel("Days from start")
    ax[0].set_ylabel("Performance")
    plt.legend()

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()
