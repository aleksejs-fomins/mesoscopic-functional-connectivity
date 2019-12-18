import numpy as np
import matplotlib.pyplot as plt


def plot_mean_variance_activity_by_learning(dataDB):
    nPlot = len(dataDB.mice)
    fig1, ax1 = plt.subplots(ncols=nPlot, figsize=(nPlot * 4, 4))
    fig2, ax2 = plt.subplots(ncols=nPlot, figsize=(nPlot * 4, 4))
    fig1.suptitle("Maximal Activity during entire day, by channel")
    fig2.suptitle("Mean Activity during entire day, by channel")

    for iPlot, mousename in enumerate(dataDB.mice):
        mouseData = dataDB.get_neuro_rows({'mousename': mousename})
        if mouseData.shape[0] > 0:
            dataIdxs = list(mouseData["date"].index)




            channelMax = np.zeros((len(dataIdxs), 12))
            channelMean = np.zeros((len(dataIdxs), 12))
            for i, dataIdx in enumerate(dataIdxs):
                dataThis = dataNeuronal[dataIdx]
                channelMax[i] = np.max(dataThis, axis=(0, 1))
                channelMean[i] = np.mean(dataThis, axis=(0, 1))

            ax1[iPlot].set_title(mousename)
            ax2[iPlot].set_title(mousename)
            ax1[iPlot].plot(channelMax)
            ax2[iPlot].plot(channelMean)
    plt.show()


def plot_performance_by_days(dataDB, outname=None, show=True):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    for mousename in dataDB.mice:
        mouseData = dataDB.get_neuro_rows({'mousename' : mousename})
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