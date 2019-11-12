import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from codes.lib.metrics.graph_lib import offdiag_idx
from codes.lib.aux_functions import merge_dicts
from codes.lib.metrics.accuracy import accuracyTests


# Create test plots from previously saved file
def fc_plots_fromfile(h5_fname, methods, logx=True, percenty=False, pTHR=None, fig_fname=None):
    with h5py.File(h5_fname, "r") as h5f:
        xparam = np.array(h5f['metadata']['xparam'])
        connTrue = np.array(h5f['metadata']['connTrue'])

        for method in methods:
            if method in h5f:
                fcData = [
                    np.array(h5f[method]['TE_table']),
                    np.array(h5f[method]['delay_table']),
                    np.array(h5f[method]['p_table'])]

                fc_plots(xparam, fcData, connTrue, method, logx=logx, percenty=percenty, pTHR=pTHR, h5_fname=None, fig_fname=fig_fname)


# Create test plots directily from data, maybe saving to h5
def fc_plots(xparam, fcData, connTrue, method, logx=True, percenty=False, pTHR=None, h5_fname=None, fig_fname=None):
    te3D, lag3D, p3D = fcData

    #####################################
    # Save data
    #####################################
    if h5_fname is not None:
        filemode = "a" if os.path.isfile(h5_fname) else "w"
        with h5py.File(h5_fname, filemode) as h5f:
            if "metadata" not in h5py:
                grp_rez = h5f.create_group("metadata")
                grp_rez['xparam'] = xparam
                grp_rez['connTrue'] = connTrue

            if method in h5py:
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

    # Delete all connections that have too high p-value, if requested
    if pTHR is not None:
        # Find which connections have high confidence
        connConf = ~np.isnan(p)
        connConf[connConf] &= p[connConf] <= pTHR

        # Set values of all other connections to NAN
        te[~connConf] = np.nan
        lag[~connConf] = np.nan
        p[~connConf] = np.nan

    # Compute statistics
    nOffDiag = nNode * (nNode - 1)
    nConnTrue = np.sum(~np.isnan(connTrue[offdiag_idx(nNode)]))
    trueRate = nConnTrue / nOffDiag

    errTestDict = merge_dicts([accuracyTests(te[:, :, i], connTrue) for i in range(nStep)])
    connFreq = nStep - np.sum(np.isnan(p), axis=2)

    #####################################
    # Plots
    #####################################

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    ax[0][0].set_title("Error rates")
    if logx:
        ax[0][0].semilogx(xparam, errTestDict["FalsePositiveRate"], '.-', label='FP_RATE')
        ax[0][0].semilogx(xparam, errTestDict["FalseNegativeRate"], '.-', label='FN_RATE')
    else:
        ax[0][0].plot(xparam, errTestDict["FalsePositiveRate"], '.-', label='FP_RATE')
        ax[0][0].plot(xparam, errTestDict["FalseNegativeRate"], '.-', label='FN_RATE')

    if trueRate > 0:
        ax[0][0].axhline(y=trueRate, linestyle='--', label='TrueRate')

    # Write x-axis as percent
    if percenty:
        ax[0][0].set_yticklabels(['{:,.2%}'.format(x) for x in ax[0][0].get_yticks()])

    ax[0][0].legend()

    ax[0][1].set_title("True connections")
    ax[0][1].imshow(connTrue)

    ax[1][1].set_title("Frequencies of connections")
    ax[1][1].imshow(connFreq, vmin=0, vmax=nStep)

    ax[1][0].set_title("TE for each connection")
    ax[1][2].set_title("p for each connection")
    ax[0][2].set_title("lag for each connection")
    ax[0][2].set_yticks(list(range(1, int(np.nanmax(lag)) + 1)))
    for i in range(nNode):
        for j in range(nNode):
            if logx:
                ax[1][0].loglog(xparam, te[i, j, :], '.-')
                ax[1][2].loglog(xparam, p[i, j, :], '.-')
                ax[0][2].semilogx(xparam, lag[i, j, :], '.-')
            else:
                ax[1][0].semilogy(xparam, te[i, j, :], '.-')
                ax[1][2].semilogy(xparam, p[i, j, :], '.-')
                ax[0][2].plot(xparam, lag[i, j, :], '.-')

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)

    plt.show()