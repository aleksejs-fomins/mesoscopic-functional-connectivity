import numpy as np
import matplotlib.pyplot as plt

from codes.lib.metrics.graph_lib import is_conn, offdiag_1D, diag_idx
from codes.lib.metrics.accuracy import accuracyTests, accuracyIndices
from codes.lib.plots.matrix import imshowAddColorBar
from codes.lib.plots.matplotblib_lib import set_percent_axis_y, hist_int

# Decide which plot function to use
plotfuncLinY = lambda ax, logx: ax.semilogx if logx else ax.plot
plotfuncLogY = lambda ax, logx: ax.loglog if logx else ax.semilogy


def setmask(arr, mask, val):
    arrNew = np.copy(arr)
    arrNew[~mask] = val
    return arrNew


def errorPlot(ax, x, yReal, yErr, label, color, logx=True):
    plotfuncLinY(ax, logx)(x, yReal, color=color, label=label)
    ax.fill_between(x, yReal-yErr, yReal+yErr, alpha=0.3, edgecolor=color, facecolor=color, antialiased=True)


def errorPlotExtended(ax, x, data, mask, logx, label, color, axis=0):
    connFreq = np.mean(np.sum(mask, axis=axis))

    if connFreq > 3:
        dataMasked = setmask(data, mask, np.nan)
        mu = np.nanmean(dataMasked, axis=axis)
        std = np.nanstd(dataMasked, axis=axis)
        errorPlot(ax, x, mu, std, label, color=color, logx=logx)
    elif connFreq > 0:
        nDataArr = np.sum(mask, axis=axis)
        xArr = np.repeat(np.array([x]).T, nDataArr)
        yArr = data[mask]
        plotfuncLinY(ax, logx)(xArr, yArr, '.', color=color, label=label)


# Create test plots directily from data, maybe saving to h5
def fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=True, percenty=False, fig_fname=None):
    #####################################
    # Analysis
    #####################################

    # Copy data to avoid modifying originals
    te, lag, p = np.copy(fcData)
    nNode, _, nStep = te.shape

    #  Find which connections have high confidence
    isConnConf     = is_conn(p, pTHR)
    isConnTrue = ~np.isnan(connTrue)
    isConnConfOffdiag1D = offdiag_1D(isConnConf)
    isConnTrueOffdiag1D = offdiag_1D(isConnTrue)

    freqTrue = np.mean(isConnTrueOffdiag1D)

    # Delete all connections that have too high p-value
    # Set values of all other connections to NAN
    te[~isConnConf] = np.nan
    lag[~isConnConf] = np.nan
    p[~isConnConf] = np.nan

    # Compute statistics
    _, errIdxsDict = accuracyIndices(isConnConfOffdiag1D, isConnTrueOffdiag1D)
    errTestDict = accuracyTests(isConnConfOffdiag1D, isConnTrueOffdiag1D)
    connFreq2DTot = np.mean(isConnConf, axis=2)
    # connFreqTPByTime = np.mean(np.sum(errIdxsDict['TP'], axis=0))
    connFreqFPByTime = np.mean(np.sum(errIdxsDict['FP'], axis=0))
    doPlotTP = freqTrue >= 0
    doPlotFP = connFreqFPByTime >= 0

    # TE statistics
    te1DoffDiag = offdiag_1D(te)
    p1DoffDiag = offdiag_1D(p)

    # Lag statistics
    minLag = 0
    maxLag = np.nanmax(lag)
    lag1DoffDiag = offdiag_1D(lag)
    if doPlotFP:
        lagsFP = lag1DoffDiag[errIdxsDict['FP']]  # Lags of FP connections
    if doPlotTP:
        lagsTP = lag1DoffDiag[errIdxsDict['TP']]  # Lags of TP connections
    # Note that for FN and TN lags don't exist, since no connections were found

    #####################################
    # Plots
    #####################################

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    ###############################
    # Error rates
    ###############################
    ax[0][0].set_title("Error frequencies")

    plotfuncLogY(ax[0][0], logx)(xparam, errTestDict["FPR"], '.-', label='FPR')
    if doPlotTP:
        plotfuncLogY(ax[0][0], logx)(xparam, errTestDict["TPR"], '.-', label='TPR')
        #ax[0][0].axhline(y=freqTrue, linestyle='--', label='P')  # Add max possible TP frequency

    if percenty:  # Write y-axis as percent
        set_percent_axis_y(ax[0][0])
    ax[0][0].legend()

    ###############################
    # Connectivity
    ###############################
    ax[0][1].set_title("True connections")
    img01 = ax[0][1].imshow(connTrue, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[0][1], img01)

    ax[1][1].set_title("Frequencies of connections")
    img11 = ax[1][1].imshow(connFreq2DTot, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[1][1], img11)

    ###############################
    # TE values
    ###############################
    ax[1][0].set_title("TE for each connection")
    # alphaTE = 0.1 if doPlotTP or doPlotFP else 1.0
    # for i in range(nNode):
    #     for j in range(nNode):
    #         plotfuncLogY(ax[1][0], logx)(xparam, te[i, j, :], '.-', alpha=alphaTE)

    errorPlotExtended(ax[1][0], xparam, te1DoffDiag, errIdxsDict['FP'], logx, "FP", color='r', axis=0)
    errorPlotExtended(ax[1][0], xparam, te1DoffDiag, errIdxsDict['TP'], logx, "TP", color='g', axis=0)
    # Note that for FN and TN p-values don't exist, since no connections were found

    ax[1][0].legend()

    ###############################
    # P-values
    ###############################
    ax[1][2].set_title("Mean p-value")
    errorPlotExtended(ax[1][2], xparam, p1DoffDiag, errIdxsDict['FP'], logx, "FP", color='r', axis=0)
    errorPlotExtended(ax[1][2], xparam, p1DoffDiag, errIdxsDict['TP'], logx, "TP", color='g', axis=0)
    ax[1][2].legend()

    ###############################
    # Lags
    ###############################
    ax[0][2].set_title("lag distribution")
    if freqTrue > 0:
        hist_int(ax[0][2], [lagsTP, lagsFP], labels=['TP', 'FP'], xmin=minLag, xmax=maxLag)
    else:
        hist_int(ax[0][2], [lagsFP], labels=['FP'], xmin=minLag, xmax=maxLag)

    ###############################
    # Save figure
    ###############################
    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


# Create type-1 plots for BTE with extra curve for false positives in the upper quadrant
def bte_accuracy_special_fromfile(xparam, fcData, connTrue, method, pTHR, connUndecided, logx=True, percenty=False, fig_fname=None):
    # Copy data to avoid modifying originals
    te, lag, p = np.copy(fcData)
    nNode, _, nStep = te.shape

    #  Find which connections have high confidence
    connExclude = np.logical_or(connUndecided, diag_idx(nNode))

    isConn1DeffH1 = is_conn(offdiag_1D(p), pTHR)
    isConn1DeffH2 = is_conn(p[~connExclude], pTHR)
    isConn1DeffH1True = ~np.isnan(offdiag_1D(connTrue))
    isConn1DeffH2True = ~np.isnan(connTrue[~connExclude])

    freqTrueH1 = np.mean(isConn1DeffH1True)
    freqTrueH2 = np.mean(isConn1DeffH2True)

    ratioH2H1 = len(isConn1DeffH2) / len(isConn1DeffH1)

    # Compute statistics
    _, errIdxsDictH1 = accuracyIndices(isConn1DeffH1, isConn1DeffH1True)
    _, errIdxsDictH2 = accuracyIndices(isConn1DeffH2, isConn1DeffH2True)
    errTestDictH1 = accuracyTests(isConn1DeffH1, isConn1DeffH1True)
    errTestDictH2 = accuracyTests(isConn1DeffH2, isConn1DeffH2True)

    isConnConf = is_conn(p, pTHR)
    te[~isConnConf] = np.nan
    te1DoffDiagH1 = offdiag_1D(te)
    te1DoffDiagH2 = te[~connExclude]

    # teFPH1 = setmask(te1DoffDiagH1, errIdxsDictH1['FP'], np.nan)
    # teFPH2 = setmask(te1DoffDiagH2, errIdxsDictH2['FP'], np.nan)
    # teTPH2 = setmask(te1DoffDiagH2, errIdxsDictH2['TP'], np.nan)

    #####################################
    # Plots
    #####################################

    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Error frequencies")

    # plotfuncLinY(ax[0], logx)(xparam, errTestDictH1["FP frequency"], '.-', label='FP-H1')
    # plotfuncLinY(ax[0], logx)(xparam, errTestDictH2["FP frequency"], '.-', label='FP-H2')
    # plotfuncLinY(ax[0], logx)(xparam, errTestDictH2["FN frequency"], '.-', label='FN')

    plotfuncLogY(ax[0], logx)(xparam, errTestDictH1["FPR"], '.-', color='r', label='FPR-H1')
    plotfuncLogY(ax[0], logx)(xparam, errTestDictH2["FPR"], '.-', color='y', label='FPR-H2')
    plotfuncLogY(ax[0], logx)(xparam, errTestDictH2["TPR"], '.-', color='g', label='TPR')
    #
    # if freqTrueH2 > 0:
    #     ax[0].axhline(y=freqTrueH2, linestyle='--', label='P')

    # Write y-axis as percent
    if percenty:
        set_percent_axis_y(ax[0])
    ax[0].legend()

    # plotfuncLinY(ax[1], logx)(xparam, np.nanmean(teFPH1, axis=0), label='FP_H1')
    # plotfuncLinY(ax[1], logx)(xparam, np.nanmean(teFPH2, axis=0), label='FP_H2')
    # plotfuncLinY(ax[1], logx)(xparam, np.nanmean(teTPH2, axis=0), label='TP_H2')

    errorPlotExtended(ax[1], xparam, te1DoffDiagH1, errIdxsDictH1['FP'], logx, "FP", color='r', axis=0)
    errorPlotExtended(ax[1], xparam, te1DoffDiagH2, errIdxsDictH2['FP'], logx, "FP", color='y', axis=0)
    errorPlotExtended(ax[1], xparam, te1DoffDiagH2, errIdxsDictH2['TP'], logx, "TP", color='g', axis=0)
    # Note that for FN and TN p-values don't exist, since no connections were found

    ax[1].legend()

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


# Plot Type 1 error as function of computation method
def fc_accuracy_plots_bymethod(xparam, fcDataDict, pTHR, logx=True, percenty=False, fig_fname=None):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Type 1 error")

    for method, fcData in fcDataDict.items():
        te, lag, p = fcData
        nNode, _, nStep = te.shape

        isConn1D = is_conn(offdiag_1D(p), pTHR)
        nConn = np.sum(isConn1D, axis=0)

        plotfuncLinY(ax, logx)(xparam, nConn, '.-', label=method)

    if percenty:
        set_percent_axis_y(ax)
    ax.legend()

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


# Would compare type 1 accuracy for every model. Currently low priority
#def fc_accuracy_plots_bymodel():


def fc_accuracy_plots_vsreal(xparam, fcDataDictSim, fcDataDictReal, pTHR, logx=True, percenty=False, fig_fname=None):
    pass