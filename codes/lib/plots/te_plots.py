import os, sys
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

import graph_lib
from signal_lib import resample

# Determine where a link is present
def is_conn(p, pTHR):
    pTMP = np.copy(p)
    pTMP[np.isnan(p)] = 100  # Set p-value of NAN connections to infinity to exclude them
    return pTMP < pTHR      # Only include connections that are likely (low p-value)

# Indices of all off-diagonal elements
def is_not_diag(N):
    return ~np.eye(N)# (1 - np.diag(np.ones(N))).astype(bool)

# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]


# Number of connections as a function of time
def plot_te_binary_metrics_bytime(outname, timesLst, dataLst, labelLst, pTHR, timestep):
    binaryMetrics = {    # 1 number per matrix
        "totalConn"               : np.sum,
        'std of in degree'        : lambda M: np.std(graph_lib.degree_in(M)),
#         'out degree'              : graph_lib.degree_out,
#         'total degree'            : graph_lib.degree_tot,
#         'reciprocal degree'       : graph_lib.degree_rec,
        'cc-total-normalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=True)),
        'cc-total-unnormalized'   : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=False)),
        'cc-in-normalized'        : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=True)),
        'cc-in-unnormalized'      : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=False)),
        'cc-out-normalized'       : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=True)),
        'cc-out-unnormalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=False))
    }
    
    nMetrics = len(binaryMetrics)
    fig, ax = plt.subplots(ncols=nMetrics, figsize=(10*nMetrics, 10))
    
    nFiles = len(dataLst)
    for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
        te, lag, p = data
        binaryConnMat = is_conn(p, pTHR)
        
        for iMetric, (metricName, metricFunc) in enumerate(binaryMetrics.items()):
            metricByTime = [metricFunc(binaryConnMat[:,:,iTime]) for iTime in len(times)]
            ax[iMetric].plot(times, metricByTime, label=label[12:17], color=((idxFile / nFiles), 0, 0))

    # Special properties for number of connections
    N_CH = te.shape[0]
    have_bte = "BivariateTE" in outname
    plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance', linewidth=2.0)
    plt.xlim(0, 10)
    plt.ylim(0, N_CH*(N_CH-1))
    
    for iMetric, (metricName, metricFunc) in enumerate(binaryMetrics.items()):
        ax[iMetric].xlabel("time, seconds")
        ax[iMetric].ylabel(metricName)
        ax[iMetric].legend()
    plt.savefig(outname)
    plt.close()
    
    
# Number of connections as a function of time
def plot_te_float_metrics_bytime(outname, timesLst, dataLst, labelLst, pTHR, timestep):        
    floatMetrics = {    # 1 number per matrix
        "maximal TE"              : np.max,
        "mean TE"                 : lambda M : np.mean(M[M > 0]),
        "total TE"                : np.sum,
        'std of in degree'        : lambda M: np.std(graph_lib.degree_in(M)),
#         'out degree'              : graph_lib.degree_out,
#         'total degree'            : graph_lib.degree_tot,
#         'reciprocal degree'       : graph_lib.degree_rec,
        'cc-total-normalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=True)),
        'cc-total-unnormalized'   : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=False)),
        'cc-in-normalized'        : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=True)),
        'cc-in-unnormalized'      : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=False)),
        'cc-out-normalized'       : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=True)),
        'cc-out-unnormalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=False))
    }
    
    
    nMetrics = len(metrics)
    nFiles = len(dataLst)
    fig, ax = plt.subplots(ncols=nMetrics, figsize=(10*nMetrics, 10))
    for iMetric, (metricName, metricFunc) in enumerate(metrics.items()):
        for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
            te, lag, p = data
            teZeroNAN = np.copy(te)
            teZeroNAN[~is_conn(p, pTHR)] = 0   # Set TE of all non-existing connections to zero

            teMetric = [metricFunc(teZeroNAN[:,:,iTime]) for iTime in range(len(times))]
            ax[iMetric].plot(times, teMetric, label=label[12:17], color=((idxFile / nFiles), 0, 0))

        ax[iMetric].xlabel("time, seconds")
        ax[iMetric].ylabel(metricName + " transfer entropy")
        ax[iMetric].legend()
    plt.savefig(outname)
    plt.close()

    
# Strategy 1: Conn=True if have at least 1 conn within range
# Strategy 2: Compute 1 metric per time step, average over range
def plot_te_binary_metrics_rangebydays(outname, timesLst, dataLst, labelLst, rangesSec, pTHR, timestep):
    binaryMetrics = {    # 1 number per time series of matrices
        "totalConn"               : np.sum,
        'std of in degree'        : lambda M: np.std(graph_lib.degree_in(M)),
#         'out degree'              : graph_lib.degree_out,
#         'total degree'            : graph_lib.degree_tot,
#         'reciprocal degree'       : graph_lib.degree_rec,
        'cc-total-normalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=True)),
        'cc-total-unnormalized'   : lambda M: np.mean(graph_lib.cl_coeff(M, kind='tot', normDegree=False)),
        'cc-in-normalized'        : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=True)),
        'cc-in-unnormalized'      : lambda M: np.mean(graph_lib.cl_coeff(M, kind='in', normDegree=False)),
        'cc-out-normalized'       : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=True)),
        'cc-out-unnormalized'     : lambda M: np.mean(graph_lib.cl_coeff(M, kind='out', normDegree=False))
    }
    
    
    fig, ax = plt.subplots(figsize=(10, 10))

    nFiles = len(dataLst)
    for rangeName, rangeSeconds in rangesSec.items():    
        atLeast1ConnPerSession = []
        for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
            rng = slice_sorted(times, rangeSeconds[0])
            
            te, lag, p = data
            pRng = p[:,:,rng[0]:rng[1]]
            atLeast1ConnPerSession += [(np.sum(is_conn(pRng, pTHR), axis=2) > 0).astype(int)]

        nConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
        sharedConn      = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax.plot(nConnPerSession, label=rangeName+'_total')
        ax.plot(sharedConn, '--', label=rangeName+'_shared')
    
    N_CH = te.shape[0]
    ax.set_ylim(0, N_CH*(N_CH-1))
    ax.set_xticks(list(range(nFiles)))
    ax.set_xticklabels([label[12:17] for label in labelLst])
    ax.legend()
    plt.savefig(outname)
    plt.close()
  
    
def plot_te_mean_tot_rangebydays(outname, timesLst, dataLst, labelLst, rangesSec, pTHR, timestep):
    metrics = {
        "mean" : np.mean,
        "tot" : np.sum}
    
    nMetrics = len(metrics)
    nFiles = len(dataLst)
    fig, ax = plt.subplots(ncols=nMetrics, figsize=(10*nMetrics, 10))
    
    for iMetric, (metricName, metricFunc) in enumerate(metrics.items()):
        for rangeName, rangeSeconds in rangesSec.items():
            teMetricArr = np.zeros(nFiles)
            for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
                rng = slice_sorted(times, rangeSeconds[0])
                te, lag, p = data
                pRng = p[:,:,rng[0]:rng[1]]

                teFlatRange = te[is_conn(pRng, pTHR)]

                teMetricArr[idxFile] = metricFunc(teFlatRange)

            ax[iMetric].plot(teMetricArr, label=rangeName)
        
        ax[iMetric].set_title(metricName + " transfer entropy")
        ax[iMetric].set_xticks(list(range(nFiles)))
        ax[iMetric].set_xticklabels([label[12:17] for label in labelLst])
        ax[iMetric].legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_nconn_rangebydays(outname, timesLst, dataLst, labelLst, rangesSec, pTHR, timestep):
    fig, ax = plt.subplots(figsize=(10, 10))

    nFiles = len(dataLst)
    for rangeName, rangeSeconds in rangesSec.items():    
        atLeast1ConnPerSession = []
        for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
            rng = slice_sorted(times, rangeSeconds[0])
            
            te, lag, p = data
            pRng = p[:,:,rng[0]:rng[1]]
            atLeast1ConnPerSession += [(np.sum(is_conn(pRng, pTHR), axis=2) > 0).astype(int)]

        nConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
        sharedConn      = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax.plot(nConnPerSession, label=rangeName+'_total')
        ax.plot(sharedConn, '--', label=rangeName+'_shared')
    
    N_CH = te.shape[0]
    ax.set_ylim(0, N_CH*(N_CH-1))
    ax.set_xticks(list(range(nFiles)))
    ax.set_xticklabels([label[12:17] for label in labelLst])
    ax.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_avgnconn_rangebydays(outname, timesLst, dataLst, labelLst, rangesSec, pTHR, timestep):
    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))

    nFiles = len(dataLst)
    for rangeName, rangeSeconds in rangesSec.items():
        freqConnRange = []
        for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
            rng = slice_sorted(times, rangeSeconds[0])
            te, lag, p = data
            pRng = p[:,:,rng[0]:rng[1]]
            
            conn1DOffDiag = is_conn(pRng, pTHR)[is_not_diag(p.shape[0])]
            freqConnRange += [np.mean(conn1DOffDiag, axis=1)]

        freqConnPerSession = [np.mean(freq) for freq in freqConnRange]
        freqSharedConn = [np.nan]
        relFreqSharedConn = [np.nan]
        for idxFile in range(1, nFiles):
            summ = freqConnRange[idxFile-1] + freqConnRange[idxFile]
            diff = freqConnRange[idxFile-1] - freqConnRange[idxFile]
            summ_L1 = np.mean(np.abs(summ))
            diff_L1 = np.mean(np.abs(diff))
            freqSharedConn    += [diff_L1]
            relFreqSharedConn += [diff_L1 / summ_L1]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax[0].plot(freqConnPerSession,  label=rangeName)
        ax[1].plot(freqSharedConn, '--', label=rangeName)
        ax[2].plot(relFreqSharedConn, '--', label=rangeName)
        
    ax[0].set_title("Average connection frequency")
    ax[1].set_title("Average connection frequency L1 distance")
    ax[2].set_title("Relative connection frequency L1 distance")
    ax[0].set_ylim([0, 0.6])
    ax[1].set_ylim([0, 0.6])
    ax[2].set_ylim([0, 1])
    ax[0].set_xticks(list(range(nFiles)))
    ax[1].set_xticks(list(range(nFiles)))
    ax[2].set_xticks(list(range(nFiles)))
    ax[0].set_xticklabels([label[12:17] for label in labelLst])
    ax[1].set_xticklabels([label[12:17] for label in labelLst])
    ax[2].set_xticklabels([label[12:17] for label in labelLst])
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_shared_link_scatter(outname, timesLst, dataLst, labelLst, rangesSec, pTHR, timestep):
    nFiles = len(dataLst)
    N_ROWS = nFiles-1
    N_COLS = len(rangesSec)
    SHARED_MAX_EFF = 0
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(5*N_COLS, 5*N_ROWS), tight_layout=True)
    
    for iRNG, (rangeName, rangeSeconds) in enumerate(rangesSec.items()):
        # Name plot
        ax[0][iRNG].set_title(rangeName)
        
        rng_len = 0
        nConnPerRng = []
        for idxFile, (times, data, label) in enumerate(zip(timesLst, dataLst, labelLst)):
            rng = slice_sorted(times, rangeSeconds[0])
            rng_len = np.max([rng_len, rng[1] - rng[0]])
            
            te, lag, p = data
            pRng = p[:,:,rng[0]:rng[1]]
            nConnPerRng += [np.sum(is_conn(pRng, pTHR), axis=2)]

        # Compute and plot number of shared connections
        sharedConn = []
        for idxFile in range(1, nFiles):
            M = np.zeros((rng_len+1, rng_len+1))
            for x, y in zip(nConnPerRng[idxFile-1].flatten(), nConnPerRng[idxFile].flatten()):
                SHARED_MAX_EFF = np.max([SHARED_MAX_EFF, x, y])
                M[x][y] += 1

            ax[idxFile-1][iRNG].imshow(np.log(M+1))
            ax[idxFile-1][iRNG].set_xlabel(label[idxFile-1][12:17])
            ax[idxFile-1][iRNG].set_ylabel(label[idxFile][12:17])
            
    for iRow in range(N_ROWS):
        for iCol in range(N_COLS):
            ax[iRow][iCol].set_xlim([0, SHARED_MAX_EFF])
            ax[iRow][iCol].set_ylim([0, SHARED_MAX_EFF])
        
    plt.savefig(outname)
    plt.close()


def plot_te_distribution(outname, dataLst, labelLst, pTHR, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(dataLst)
    for idxFile, (data, label) in enumerate(zip(dataLst, labelLst)):
        te, lag, p = data
        teflat = te[is_conn(p, pTHR)]
        
        plt.hist(teflat, bins='auto', label=label[12:17], color=((idxFile / nFiles), 0, 0), alpha=0.3)

    plt.xscale("log")
    plt.xlim([1.0e-3,1])
    plt.xlabel("TE")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    

def plot_te_distribution_rangebydays(outname, dataLst, labelLst, rangesSec, pTHR, timestep):
    fig, ax = plt.subplots(figsize=(10, 10))

    nFiles = len(dataLst)
    for rangeName, rangeSeconds in rangesSec.items():
        rng_min = int(rangeSeconds[0] / timestep)
        rng_max = int(rangeSeconds[1] / timestep)
    
        meanTEByDay = []
        for idxFile, (data, label) in enumerate(zip(dataLst, labelLst)):
            te, lag, p = data
            pRng = p[:,:,rng_min:rng_max]
            te_rng = te[:,:,rng_min:rng_max]
            idx_conn = is_conn(pRng, pTHR)
            meanTEByDay += [np.mean(te_rng[idx_conn.astype(bool)])]

        #TODO - extrapolate
        ax.plot(meanTEByDay, label=rangeName)
    
    ax.set_xticks(list(range(nFiles)))
    ax.set_xticklabels([label[12:17] for label in labelLst])
    ax.legend()
    plt.savefig(outname)
    plt.close()
    
    

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
def plot_te_rel_diff_nlinks_bydays(outname, timesLst1, timesLst2, dataLst1, dataLst2, labelLst1, labelLst2, pTHR, timestep1, timestep2):

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
        label1 = labelLst1[idxFile]
        label2 = labelLst2[idxFile]
        assert label1 == label2, "Basenames not aligned"
        te1, lag1, p1 = dataLst1[idxFile]
        te2, lag2, p2 = dataLst2[idxFile]
        
        # Select off-diagonal elements. Mark all non-connections with 0
        is_conn1 = is_conn(p1, pTHR)
        is_conn2 = is_conn(p2, pTHR)
        te1_nonan = np.zeros(te1.shape)
        te2_nonan = np.zeros(te2.shape)
        te1_nonan[is_conn1] = te1[is_conn1]
        te2_nonan[is_conn2] = te2[is_conn2]
        te1_flat = te1_nonan[is_not_diag(te1.shape[0])]
        te2_flat = te2_nonan[is_not_diag(te2.shape[0])]
        
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
    ax.set_xticklabels([label[12:17] for label in labelLst1])
    ax.set_ylabel("Relative diff in TE")
    plt.savefig(outname)
    plt.close()
    
    
