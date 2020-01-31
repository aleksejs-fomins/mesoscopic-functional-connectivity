import numpy as np
import matplotlib.pyplot as plt

from codes.lib.aux_functions import slice_sorted
from codes.lib.stat import graph_lib
from codes.lib.signal_lib import resample, resample_shortest_linear


def _get_binary_conn(data, pTHR):
    return graph_lib.is_conn(data[2], pTHR)


def _get_fc_from_data(data, pTHR):
    teZeroNAN = np.copy(data[0])
    teZeroNAN[~graph_lib.is_conn(p, pTHR)] = 0  # Set TE of all non-existing connections to zero
    return teZeroNAN


def _get_binary_metric_dict():
    return {
        "totalConn"                 : np.sum,
        'std of in degree'          : lambda M: np.std(graph_lib.degree_in(M)),
        #         'out degree'              : graph_lib.degree_out,
        #         'total degree'            : graph_lib.degree_tot,
        #         'reciprocal degree'       : graph_lib.degree_rec,
        'cc-total-normalized'       : lambda M: np.mean(
            graph_lib.clustering_coefficient(M, kind='tot', normDegree=True)),
        'cc-total-unnormalized'     : lambda M: np.mean(
            graph_lib.clustering_coefficient(M, kind='tot', normDegree=False)),
        'cc-in-normalized'          : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='in', normDegree=True)),
        'cc-in-unnormalized'        : lambda M: np.mean(
            graph_lib.clustering_coefficient(M, kind='in', normDegree=False)),
        'cc-out-normalized'         : lambda M: np.mean(
            graph_lib.clustering_coefficient(M, kind='out', normDegree=True)),
        'cc-out-unnormalized'       : lambda M: np.mean(
            graph_lib.clustering_coefficient(M, kind='out', normDegree=False))
    }


def _get_fc_metric_dict():
    return {
        "maximal TE"              : np.max,
        "mean TE"                 : lambda M : np.mean(M[M > 0]),
        "total TE"                : np.sum,
        'std of in degree'        : lambda M: np.std(graph_lib.degree_in(M)),
#         'out degree'              : graph_lib.degree_out,
#         'total degree'            : graph_lib.degree_tot,
#         'reciprocal degree'       : graph_lib.degree_rec,
        'cc-total-normalized'     : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='tot', normDegree=True)),
        'cc-total-unnormalized'   : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='tot', normDegree=False)),
        'cc-in-normalized'        : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='in', normDegree=True)),
        'cc-in-unnormalized'      : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='in', normDegree=False)),
        'cc-out-normalized'       : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='out', normDegree=True)),
        'cc-out-unnormalized'     : lambda M: np.mean(graph_lib.clustering_coefficient(M, kind='out', normDegree=False))
    }


def _plot_generic_vs_time(timesLst, dataLst, metricDict, dataFunc, pTHR):
    nMetrics = len(metricDict)
    nFiles = len(dataLst)

    # Metrics by time
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(ncols=nMetrics, figsize=(10*nMetrics, 10))
    for iMetric, (metricName, metricFunc) in enumerate(metricDict.items()):

        # Note: Some files are a bit shorter than others. We truncate time to the shortest
        metricValLst = []
        for iFile, (times, data) in enumerate(zip(timesLst, dataLst)):
            nTimes = len(times)
            trgData = dataFunc(data, pTHR)
            metricValLst += [[metricFunc(trgData[:, :, iTime]) for iTime in range(nTimes)]]

        timesResampled, metricResampled2D = resample_shortest_linear(timesLst, metricValLst)

        metricMean = np.mean(metricResampled2D, axis=0)
        metricStd = np.std(metricResampled2D, axis=0)

        ax[iMetric].plot(timesResampled, metricMean)
        ax[iMetric].fill_between(timesResampled, metricMean - metricStd, metricMean + metricStd, alpha=0.3)
        ax[iMetric].set_title("nPoints=" + str(nFiles))
        ax[iMetric].set_xlabel("time, seconds")
        ax[iMetric].set_ylabel(metricName)

    return fig, ax


# Metrics for binary connectivity matrix as function of time
def plot_fc_binary_vs_time(outnameBase, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
    for rangeLabel, timeRangeLst, dataRangeLst in zip(rangeLabelLst, timesLst, dataLst):
        outname = outnameBase[:-4] + "_" + rangeLabel + outnameBase[-4:]

        # Metrics by time
        fig, ax = _plot_generic_vs_time(timeRangeLst, dataRangeLst, _get_binary_metric_dict(), _get_binary_conn, pTHR)

        # Special properties for number of connections
        nChannel = dataLst[0][0].shape[0]
        have_bte = "BivariateTE" in outname
        ax[0].axhline(y=4.0 * nChannel / 12 if have_bte else 1.0 * nChannel / 12, linestyle="--", label='chance', linewidth=2.0)
        ax[0].set_xlim(0, 10)
        ax[0].set_ylim(0, nChannel*(nChannel-1))

        # ax[iMetric].plot(times, metricByTime, label=label[12:17], color=((iFile / nFiles), 0, 0))
        # ax[iMetric].legend()
        #print("Saving figure to", outname)
        plt.savefig(outname)
        plt.close()


# Metrics for TE matrix as function of time
def plot_fc_mag_vs_time(outnameBase, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
    for rangeLabel, timeRangeLst, dataRangeLst in zip(rangeLabelLst, timesLst, dataLst):
        outname = outnameBase[:-4] + "_" + rangeLabel + outnameBase[-4:]

        # Plot
        fig, ax = _plot_generic_vs_time(timeRangeLst, dataRangeLst, _get_fc_metric_dict(), _get_fc_from_data, pTHR)

        plt.savefig(outname)
        plt.close()


# Metrics for binary connectivity matrix as function of days
#   Strategy 1: Conn=True if have at least 1 conn within range
#   Strategy 2: Compute 1 metric per time step, average over range
# TODO - vertical line for performance
# TODO - extrapolate simulated result T1 nconn
def plot_fc_binary_vs_days(outname, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
    binaryMetrics = _get_binary_metric_dict()
    nMetrics = len(binaryMetrics)
    nFiles = len(dataLabelLst)
    
    fig, ax = plt.subplots(nrows=2, ncols=nMetrics, figsize=(10*nMetrics, 2*10))

    for rangeLabel, timeRangeLst, dataRangeLst in zip(rangeLabelLst, timesLst, dataLst):
        metricArrStrat1    = np.zeros((nMetrics, nFiles))
        metricArrStrat2mu  = np.zeros((nMetrics, nFiles))
        metricArrStrat2std = np.zeros((nMetrics, nFiles))
    
        # Compute metrics
        for iFile, (times, data) in enumerate(zip(timeRangeLst, dataRangeLst)):
            binaryConnMatRng = _get_binary_conn(data, pTHR)
            binaryConnMatRngAtLeast1 = np.max(binaryConnMatRng, axis = 2)

            for iMetric, (metricName, metricFunc) in enumerate(binaryMetrics.items()):
                metricStrat2arr = np.array([metricFunc(binaryConnMatRng[:,:,iTime]) for iTime in range(binaryConnMatRng.shape[2])])
                metricArrStrat1[iMetric, iFile]   = metricFunc(binaryConnMatRngAtLeast1)
                metricArrStrat2mu[iMetric, iFile] = np.mean(metricStrat2arr)
                metricArrStrat2std[iMetric, iFile] = np.std(metricStrat2arr)

        # Plot metrics
        for iMetric in range(nMetrics):
            pltx     = np.arange(nFiles)
            plt1y    = metricArrStrat1[iMetric]
            plt2y    = metricArrStrat2mu[iMetric]
            plt2ymin = metricArrStrat2mu[iMetric] - metricArrStrat2std[iMetric]
            plt2ymax = metricArrStrat2mu[iMetric] + metricArrStrat2std[iMetric]

            ax[0, iMetric].plot(plt1y, label=rangeLabel)
            ax[1, iMetric].plot(pltx, plt2y, label=rangeLabel)
            ax[1, iMetric].fill_between(pltx, plt2ymin, plt2ymax, alpha=0.3)#, label=rangeName)

    # Set labels on axis
    nChannel = dataLst[0][0].shape[0]
    for iStrat in range(2):
        ax[iStrat, 0].set_ylim(0, nChannel*(nChannel-1))
        for iMetric, metricName in enumerate(binaryMetrics.keys()):
            ax[iStrat, iMetric].set_ylabel(metricName)
            ax[iStrat, iMetric].set_xticks(list(range(nFiles)))
            ax[iStrat, iMetric].set_xticklabels([label[12:17] for label in dataLabelLst], rotation='vertical')
            ax[iStrat, iMetric].legend()
    plt.savefig(outname)
    plt.close()


# Metrics for binary connectivity matrix as function of days
#   Strategy 1: Conn=True if have at least 1 conn within range
#   Strategy 2: Compute 1 metric per time step, average over range
def plot_fc_mag_vs_days(outname, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
    floatMetrics = _get_fc_metric_dict()
    nMetrics = len(floatMetrics)
    nFiles = len(dataLabelLst)

    fig, ax = plt.subplots(ncols=nMetrics, figsize=(10 * nMetrics, 10))

    for rangeLabel, timeRangeLst, dataRangeLst in zip(rangeLabelLst, timesLst, dataLst):
        metricArrMu = np.zeros((nMetrics, nFiles))
        metricArrStd = np.zeros((nMetrics, nFiles))

        # Compute metrics
        for iFile, (times, data) in enumerate(zip(timeRangeLst, dataRangeLst)):
            teRngZeroNAN = _get_fc_from_data(data, pTHR)

            for iMetric, (metricName, metricFunc) in enumerate(floatMetrics.items()):
                tmp = [metricFunc(teRngZeroNAN[:, :, iTime]) for iTime in range(teRngZeroNAN.shape[2])]
                metricArrMu[iMetric, iFile] = np.mean(tmp)
                metricArrStd[iMetric, iFile] = np.std(tmp)

        # Plot metrics
        for iMetric in range(nMetrics):
            pltx = np.arange(nFiles)
            plty = metricArrMu[iMetric]
            pltymin = metricArrMu[iMetric] - metricArrStd[iMetric]
            pltymax = metricArrMu[iMetric] + metricArrStd[iMetric]

            ax[iMetric].plot(pltx, plty, label=rangeLabel)
            ax[iMetric].fill_between(pltx, pltymin, pltymax, alpha=0.3)#, label=rangeName)

    # Set ticks for x-axis
    for iMetric, (metricName, metricFunc) in enumerate(floatMetrics.items()):
        ax[iMetric].set_ylabel(metricName + " transfer entropy")
        ax[iMetric].set_xticks(list(range(nFiles)))
        ax[iMetric].set_xticklabels([label[12:17] for label in dataLabelLst], rotation='vertical')
        ax[iMetric].legend()
    plt.savefig(outname)
    plt.close()


# def plot_te_nconn_rangebydays(outname, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
#     nFiles = len(dataLabelLst)
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     for rangeLabel, dataRangeLst in zip(rangeLabelLst, dataLst):
#         atLeast1ConnPerSession = []
#         for idxFile, data in enumerate(dataLst):
#             binaryConnMatRng = _get_binary_conn(data, pTHR)
#             atLeast1ConnPerSession += [(np.sum(binaryConnMatRng, axis=2) > 0).astype(int)]
#
#         nConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
#         sharedConn      = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]
#
#         #TODO - extrapolate
#         #have_bte = "BivariateTE" in outname
#         #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
#         ax.plot(nConnPerSession, label=rangeLabel+'_total')
#         ax.plot(sharedConn, '--', label=rangeLabel+'_shared')
#
#     nChannel = dataLst[0][0].shape[0]
#     ax.set_ylim(0, nChannel*(nChannel-1))
#     ax.set_xticks(list(range(nFiles)))
#     ax.set_xticklabels([label[12:17] for label in dataLabelLst])
#     ax.legend()
#     plt.savefig(outname)
#     plt.close()
    
    
def plot_fc_binary_avg_vs_time(outname, timesLst, dataLst, dataLabelLst, rangeLabelLst, pTHR):
    nChannel = dataLst[0][0].shape[0]

    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
    for rangeLabel, dataRangeLst in zip(rangeLabelLst, dataLst):
        freqConnRange = []
        labelsFiltered = []
        for idxFile, (data, label) in enumerate(zip(dataLst, dataLabelLst)):
            if nChannel in (12, 48):
                labelsFiltered += [label]
                binaryConnMat1D = graph_lib.offdiag_1D(_get_binary_conn(data, pTHR))
                freqConnRange += [np.mean(binaryConnMat1D, axis=1)]

        freqConnPerSession = [np.mean(freq) for freq in freqConnRange]
        freqSharedConn = [np.nan]
        relFreqSharedConn = [np.nan]
        for idxFile in range(1, len(freqConnRange)):
            summ = freqConnRange[idxFile-1] + freqConnRange[idxFile]
            diff = freqConnRange[idxFile-1] - freqConnRange[idxFile]
            summ_L1 = np.mean(np.abs(summ))
            diff_L1 = np.mean(np.abs(diff))
            freqSharedConn    += [diff_L1]
            relFreqSharedConn += [diff_L1 / summ_L1]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax[0].plot(freqConnPerSession,  label=rangeLabel)
        ax[1].plot(freqSharedConn, '--', label=rangeLabel)
        ax[2].plot(relFreqSharedConn, '--', label=rangeLabel)
        
    ax[0].set_title("Average connection frequency")
    ax[1].set_title("Average connection frequency L1 distance")
    ax[2].set_title("Relative connection frequency L1 distance")
    ax[0].set_ylim([0, 0.6])
    ax[1].set_ylim([0, 0.6])
    ax[2].set_ylim([0, 1])
    labelIdxs = list(range(len(labelsFiltered)))
    ax[0].set_xticks(labelIdxs)
    ax[1].set_xticks(labelIdxs)
    ax[2].set_xticks(labelIdxs)
    ax[0].set_xticklabels([label[12:17] for label in labelsFiltered])
    ax[1].set_xticklabels([label[12:17] for label in labelsFiltered])
    ax[2].set_xticklabels([label[12:17] for label in labelsFiltered])
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(outname)
    plt.close()



def plot_fc_vs_performance(outname, timesLst, dataLst, dataLabelLst, rangesSec, pTHR, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(dataLst)
    for idxFile, (data, label) in enumerate(zip(dataLst, dataLabelLst)):
        te, lag, p = data
        teflat = te[graph_lib.is_conn(p, pTHR)]
        
        plt.hist(teflat, bins='auto', label=label[12:17], color=((idxFile / nFiles), 0, 0), alpha=0.3)

    plt.xscale("log")
    plt.xlim([1.0e-3,1])
    plt.xlabel("TE")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    

# def plot_te_distribution_rangebydays(outname, timesLst, dataLst, dataLabelLst, rangesSec, pTHR, timestep):
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     nFiles = len(dataLst)
#     for rangeName, rangeSeconds in rangesSec.items():
#         rng_min = int(rangeSeconds[0] / timestep)
#         rng_max = int(rangeSeconds[1] / timestep)
#
#         meanTEByDay = []
#         for idxFile, (data, label) in enumerate(zip(dataLst, dataLabelLst)):
#             te, lag, p = data
#             pRng = p[:,:,rng_min:rng_max]
#             te_rng = te[:,:,rng_min:rng_max]
#             idx_conn = graph_lib.is_conn(pRng, pTHR)
#             meanTEByDay += [np.mean(te_rng[idx_conn.astype(bool)])]
#
#         #TODO - extrapolate
#         ax.plot(meanTEByDay, label=rangeName)
#
#     ax.set_xticks(list(range(nFiles)))
#     ax.set_xticklabels([label[12:17] for label in dataLabelLst])
#     ax.legend()
#     plt.savefig(outname)
#     plt.close()



'''
Task: Compute difference between connectivity predictions made by different estimators
1) Compute true    over entire trace REL(A,B),      REL(A,B) = |A - B| / |A + B|
2) Compute shuffle over entire trace REL(A, perm(B))
3) Plot with z-score

If have downsample:
- Upsample the lower frequency signal using LINEAR interpolation. The conjecture in downsampled signal is that the value does not change much during timestep interval. Using nonlinear interpolation might cause signal to go negative due to Runge effect, better not. Downsampling also does not make sense, since the faster signal does not make the assumption that it changes slowly, so its downsampled version need not make sense at all.

TODO:
* Crop beginning of window by max delta
'''
def plot_te_rel_diff_nlinks_bydays(outname, timesLst1, timesLst2, dataLst1, dataLst2, dataLabelLst1, dataLabelLst2, pTHR, timestep1, timestep2):

    # Resample two datasets to same sample times within provided range
    def times_match(times1, times2, data1, data2, t_range, RESAMPLE_PARAM):
        # The first dataset is the closest fit to the provided time range
        idx1 = slice_sorted(times1, t_range)
        times_eff = times1[idx1[0]:idx1[1]]
        data1_new = data1[:, idx1[0]:idx1[1]]
        
        # The second dataset is resampled to match the first one
        data2_new = np.array([resample(times2, d2, times_eff, RESAMPLE_PARAM) for d2 in data2])
        
        return times_eff, data1_new, data2_new
    
    nFiles1 = len(dataLst1)
    nFiles2 = len(dataLst2)
    assert nFiles1 == nFiles2, "Not the same number of file types"
    
    div = np.zeros(nFiles1)
    for idxFile in range(nFiles1):
        times1 = timesLst1[idxFile]
        times2 = timesLst2[idxFile]
        label1 = dataLabelLst1[idxFile]
        label2 = dataLabelLst2[idxFile]
        assert label1 == label2, "Basenames not aligned"
        te1, lag1, p1 = dataLst1[idxFile]
        te2, lag2, p2 = dataLst2[idxFile]
        
        # Select off-diagonal elements. Mark all non-connections with 0
        is_conn1 = graph_lib.is_conn(p1, pTHR)
        is_conn2 = graph_lib.is_conn(p2, pTHR)
        te1_nonan = np.zeros(te1.shape)
        te2_nonan = np.zeros(te2.shape)
        te1_nonan[is_conn1] = te1[is_conn1]
        te2_nonan[is_conn2] = te2[is_conn2]
        te1_flat = te1_nonan[graph_lib.offdiag_idx(te1.shape[0])]
        te2_flat = te2_nonan[graph_lib.offdiag_idx(te2.shape[0])]
        
        t_range = [
            np.max([times1[0], times2[0]]),
            np.min([times1[-1], times2[-1]])]
        
        # Resample the dataset which has the largest timestep
        # If timesteps equal, resample the 2nd one (no particular reason)
        RESAMPLE_PARAM = {'method' : 'interpolative', 'kind' : 'linear'}
        if timestep1 < timestep2:
            times_eff, te1_eff, te2_eff = times_match(times1, times2, te1_flat, te2_flat, t_range, RESAMPLE_PARAM)
        else:
            times_eff, te2_eff, te1_eff = times_match(times2, times1, te2_flat, te1_flat, t_range, RESAMPLE_PARAM)
            
        diff = np.linalg.norm(te1_eff - te2_eff)
        summ = np.linalg.norm(te1_eff + te2_eff)
        div[idxFile] = diff / summ
        
        # div = np.zeros(len(times1))
        # idxNonZero = summ != 0
        # div[idxNonZero] = diff[idxNonZero] / summ[idxNonZero]
        
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(div)
    ax.set_ylim([0,1])
    ax.set_xticks(list(range(nFiles1)))
    ax.set_xticklabels([label[12:17] for label in dataLabelLst1])
    ax.set_ylabel("Relative diff in TE")
    plt.savefig(outname)
    plt.close()
