import os, sys
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from signal_lib import resample

# Determine where a link is present
def is_conn(p, p_thr):
    p_tmp = np.copy(p)
    p_tmp[np.isnan(p)] = 100  # Set p-value of NAN connections to infinity to exclude them
    return p_tmp < p_thr      # Only include connections that are likely (low p-value)

# Indices of all off-diagonal elements
def is_not_diag(N):
    return (1 - np.diag(np.ones(N))).astype(bool)

# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]


def plot_te_nconn_bytime(outname, times_lst, data_lst, label_lst, p_thr, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (times, data, label) in enumerate(zip(times_lst, data_lst, label_lst)):
        te, lag, p = data
        
        totalConnPerTime = np.sum(is_conn(p, p_thr), axis=(0,1))
        
        plt.plot(times, totalConnPerTime, label=label[12:17], color=((idxFile / nFiles), 0, 0))

        
    N_CH = te.shape[0]
    have_bte = "BivariateTE" in outname
    plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance', linewidth=2.0)
    plt.xlim(0, 10)
    plt.ylim(0, N_CH*(N_CH-1))
    plt.xlabel("time, seconds")
    plt.ylabel("Number of connections")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_nconn_rangebydays(outname, times_lst, data_lst, label_lst, ranges_sec, p_thr, timestep):
    fig, ax = plt.subplots(figsize=(10, 10))

    nFiles = len(data_lst)
    for rng_name, rng_sec in ranges_sec.items():    
        atLeast1ConnPerSession = []
        for idxFile, (times, data, label) in enumerate(zip(times_lst, data_lst, label_lst)):
            rng = slice_sorted(times, rng_sec[0])
            
            te, lag, p = data
            p_rng = p[:,:,rng[0]:rng[1]]
            atLeast1ConnPerSession += [(np.sum(is_conn(p_rng, p_thr), axis=2) > 0).astype(int)]

        nConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
        sharedConn      = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax.plot(nConnPerSession, label=rng_name+'_total')
        ax.plot(sharedConn, '--', label=rng_name+'_shared')
    
    N_CH = te.shape[0]
    ax.set_ylim(0, N_CH*(N_CH-1))
    ax.set_xticks(list(range(nFiles)))
    ax.set_xticklabels([label[12:17] for label in label_lst])
    ax.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_avgnconn_rangebydays(outname, times_lst, data_lst, label_lst, ranges_sec, p_thr, timestep):
    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))

    nFiles = len(data_lst)
    for rng_name, rng_sec in ranges_sec.items():
        freqConnRange = []
        for idxFile, (times, data, label) in enumerate(zip(times_lst, data_lst, label_lst)):
            rng = slice_sorted(times, rng_sec[0])
            te, lag, p = data
            p_rng = p[:,:,rng[0]:rng[1]]
            
            conn_1D_offdiag = is_conn(p_rng, p_thr)[is_not_diag(p.shape[0])]
            freqConnRange += [np.mean(conn_1D_offdiag, axis=1)]

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
        ax[0].plot(freqConnPerSession,  label=rng_name)
        ax[1].plot(freqSharedConn, '--', label=rng_name)
        ax[2].plot(relFreqSharedConn, '--', label=rng_name)
        
    ax[0].set_title("Average connection frequency")
    ax[1].set_title("Average connection frequency L1 distance")
    ax[2].set_title("Relative connection frequency L1 distance")
    ax[0].set_ylim([0, 0.6])
    ax[1].set_ylim([0, 0.6])
    ax[2].set_ylim([0, 1])
    ax[0].set_xticks(list(range(nFiles)))
    ax[1].set_xticks(list(range(nFiles)))
    ax[2].set_xticks(list(range(nFiles)))
    ax[0].set_xticklabels([label[12:17] for label in label_lst])
    ax[1].set_xticklabels([label[12:17] for label in label_lst])
    ax[2].set_xticklabels([label[12:17] for label in label_lst])
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_shared_link_scatter(outname, times_lst, data_lst, label_lst, ranges_sec, p_thr, timestep):
    nFiles = len(data_lst)
    N_ROWS = nFiles-1
    N_COLS = len(ranges_sec)
    SHARED_MAX_EFF = 0
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(5*N_COLS, 5*N_ROWS), tight_layout=True)
    
    for iRNG, (rng_name, rng_sec) in enumerate(ranges_sec.items()):
        # Name plot
        ax[0][iRNG].set_title(rng_name)
        
        rng_len = 0
        nConnPerRng = []
        for idxFile, (times, data, label) in enumerate(zip(times_lst, data_lst, label_lst)):
            rng = slice_sorted(times, rng_sec[0])
            rng_len = np.max([rng_len, rng[1] - rng[0]])
            
            te, lag, p = data
            p_rng = p[:,:,rng[0]:rng[1]]
            nConnPerRng += [np.sum(is_conn(p_rng, p_thr), axis=2)]

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


def plot_te_distribution(outname, data_lst, label_lst, P_THR, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data
        teflat = te[is_conn(p, P_THR)]
        
        plt.hist(teflat, bins='auto', label=label[12:17], color=((idxFile / nFiles), 0, 0), alpha=0.3)

    plt.xscale("log")
    plt.xlim([1.0e-3,1])
    plt.xlabel("TE")
    plt.ylabel("Count")
    plt.legend()
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
def plot_te_rel_diff_nlinks_bydays(outname, times_lst1, times_lst2, data_lst1, data_lst2, label_lst1, label_lst2, P_THR, timestep1, timestep2):

    # Resample two datasets to same sample times within provided range
    def times_match(times1, times2, data1, data2, t_range, RESAMPLE_PARAM):
        # The first dataset is the closest fit to the provided time range
        idx1 = slice_sorted(times1, t_range)
        times_eff = times1[idx1[0]:idx1[1]]
        data1_new = data1[:, idx1[0]:idx1[1]]
        
        # The second dataset is resampled to match the first one
        data2_new = np.array([resample(times2, d2, times_eff, RESAMPLE_PARAM) for d2 in data2])
        
        return times_eff, data1_new, data2_new
    
    nFiles1 = len(data_lst1)
    nFiles2 = len(data_lst2)
    assert nFiles1 == nFiles2, "Not the same number of file types"
    
    div = np.zeros(nFiles1)
    for idxFile in range(nFiles1):
        times1 = times_lst1[idxFile]
        times2 = times_lst2[idxFile]
        label1 = label_lst1[idxFile]
        label2 = label_lst2[idxFile]
        assert label1 == label2, "Basenames not aligned"
        te1, lag1, p1 = data_lst1[idxFile]
        te2, lag2, p2 = data_lst2[idxFile]
        
        # Select off-diagonal elements. Mark all non-connections with 0
        is_conn1 = is_conn(p1, P_THR)
        is_conn2 = is_conn(p2, P_THR)
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
    ax.set_xticklabels([label[12:17] for label in label_lst1])
    ax.set_ylabel("Relative diff in TE")
    plt.savefig(outname)
    plt.close()
    
    
