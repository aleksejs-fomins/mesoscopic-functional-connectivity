import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from codes.lib.metrics.graph_lib import is_conn, offdiag_1D
from codes.lib.metrics.accuracy import accuracyTests, accuracyIndices
from codes.lib.plots.matrix import imshowAddColorBar
from codes.lib.plots.matplotblib_lib import set_percent_axis_y, hist_int


# Create test plots from previously saved file
def fc_accuracy_plots_fromfile(h5_fname, methods, pTHR, logx=True, percenty=False, fig_fname=None):
    with h5py.File(h5_fname, "r") as h5f:
        xparam = np.copy(h5f['metadata']['xparam'])
        connTrue = np.copy(h5f['metadata']['connTrue'])

        for method in methods:
            if method in h5f:
                fcData = [
                    np.copy(h5f[method]['TE_table']),
                    np.copy(h5f[method]['delay_table']),
                    np.copy(h5f[method]['p_table'])]

                # If figure requested, add method to its name, otherwise keep it as None
                fig_fname_method = fig_fname[:-3] + "_" + method + fig_fname[-3:] if fig_fname is not None else None

                fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=logx, percenty=percenty, h5_fname=None, fig_fname=fig_fname_method)


# Create test plots directily from data, maybe saving to h5
def fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=True, percenty=False, h5_fname=None, fig_fname=None):
    te3D, lag3D, p3D = fcData

    #####################################
    # Save data
    #####################################
    if h5_fname is not None:
        filemode = "a" if os.path.isfile(h5_fname) else "w"
        with h5py.File(h5_fname, filemode) as h5f:
            if "metadata" not in h5f.keys():
                grp_rez = h5f.create_group("metadata")
                grp_rez['xparam'] = xparam
                grp_rez['connTrue'] = connTrue

            if method in h5f.keys():
                raise ValueError("Already have data for method", method)

            grp_method = h5f.create_group(method)
            grp_method['TE_table'] = te3D
            grp_method['delay_table'] = lag3D
            grp_method['p_table'] = p3D

    #####################################
    # Analysis
    #####################################

    # Copy data to avoid modifying originals
    te, lag, p = np.copy(te3D), np.copy(lag3D), np.copy(p3D)
    nNode, _, nStep = te.shape

    #  Find which connections have high confidence
    isConnConf     = is_conn(p, pTHR)
    isConnConfTrue = ~np.isnan(connTrue)
    isConnConfOffdiag1D     = offdiag_1D(isConnConf)
    isConnConfOffdiag1DTrue = offdiag_1D(isConnConfTrue)

    freqTrue = np.mean(isConnConfOffdiag1DTrue)

    # Delete all connections that have too high p-value
    # Set values of all other connections to NAN
    te[~isConnConf] = np.nan
    lag[~isConnConf] = np.nan
    p[~isConnConf] = np.nan

    # Compute statistics
    errIdxsDict = accuracyIndices(isConnConfOffdiag1D, isConnConfOffdiag1DTrue)
    errTestDict = accuracyTests(isConnConfOffdiag1D, isConnConfOffdiag1DTrue)
    connFreq = np.mean(isConnConf, axis=2)

    # Lag statistics
    minLag = 0
    maxLag = np.nanmax(lag)
    lag1DoffDiag = offdiag_1D(lag)
    lagsTP = lag1DoffDiag[errIdxsDict['TP']]  # Lags of TP connections
    lagsFP = lag1DoffDiag[errIdxsDict['FP']]  # Lags of FP connections
    # Note that for FN and TN lags don't exist, since no connections were found

    # P-value statistics
    p1DoffDiag = offdiag_1D(p)
    pvalTP = p1DoffDiag[errIdxsDict['TP']]  # Lags of TP connections
    pvalFP = p1DoffDiag[errIdxsDict['FP']]  # Lags of FP connections
    # Note that for FN and TN p-values don't exist, since no connections were found

    #####################################
    # Plots
    #####################################

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    ax[0][0].set_title("Error frequencies")
    if logx:
        ax[0][0].semilogx(xparam, errTestDict["FP frequency"], '.-', label='FP')
        ax[0][0].semilogx(xparam, errTestDict["FN frequency"], '.-', label='FN')
    else:
        ax[0][0].plot(xparam, errTestDict["FP frequency"], '.-', label='FP')
        ax[0][0].plot(xparam, errTestDict["FN frequency"], '.-', label='FN')

    if freqTrue > 0:
        ax[0][0].axhline(y=freqTrue, linestyle='--', label='P')

    # Write y-axis as percent
    if percenty:
        set_percent_axis_y(ax[0][0])
    ax[0][0].legend()

    ax[0][1].set_title("True connections")
    img01 = ax[0][1].imshow(connTrue, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[0][1], img01)

    ax[1][1].set_title("Frequencies of connections")
    img11 = ax[1][1].imshow(connFreq, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[1][1], img11)

    ax[1][0].set_title("TE for each connection")
    ax[1][2].set_title("p for each connection")

    for i in range(nNode):
        for j in range(nNode):
            if logx:
                ax[1][0].loglog(xparam, te[i, j, :], '.-')
                ax[1][2].loglog(xparam, p[i, j, :], '.-')
            else:
                ax[1][0].semilogy(xparam, te[i, j, :], '.-')
                ax[1][2].semilogy(xparam, p[i, j, :], '.-')

    # Make histogram
    ax[0][2].set_title("lag distribution")
    hist_int(ax[0][2], [lagsTP, lagsFP], labels=['TP', 'FP'], xmin=minLag, xmax=maxLag)

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)

    #plt.show()