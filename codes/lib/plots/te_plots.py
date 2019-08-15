import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from signal_lib import resample

# Determine where a link is present
def is_conn(p, p_thr):
    p_tmp = np.copy(p)
    p_tmp[np.isnan(p)] = 100            # Set p-value of NAN connections to infinity to exclude them
    return (p_tmp < p_thr).astype(int)  # Only include connections that are likely (low p-value)
    

def plot_te_nconn_bytime(outname, data_lst, label_lst, p_thr, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data

        times = timestep * np.linspace(0, te.shape[2], te.shape[2])
        totalConnPerTime = [np.sum(is_conn(p[:,:,i], p_thr)) for i in range(te.shape[2])]
        
        plt.plot(times, totalConnPerTime, label=label[12:17], color=((idxFile / nFiles), 0, 0))

    have_bte = "BivariateTE" in outname
    plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
    plt.xlabel("time, seconds")
    plt.ylabel("Number of connections")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_nconn_rangebydays(outname, data_lst, label_lst, ranges_sec, p_thr, timestep):
    fig, ax = plt.subplots(figsize=(10, 10))

    nFiles = len(data_lst)
    for rng_name, rng_sec in ranges_sec.items():
        rng_min = int(rng_sec[0] / timestep)
        rng_max = int(rng_sec[1] / timestep)
    
        atLeast1ConnPerSession = []
        for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
            te, lag, p = data
            p_rng = p[:,:,rng_min:rng_max]
            atLeast1ConnPerSession += [(np.sum(is_conn(p_rng, p_thr), axis=2) > 0).astype(int)]

        nConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
        sharedConn = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax.plot(nConnPerSession, label=rng_name+'_total')
        ax.plot(sharedConn, '--', label=rng_name+'_shared')
    
    ax.set_xticks(list(range(nFiles)))
    ax.set_xticklabels([label[12:17] for label in label_lst])
    ax.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_avgnconn_rangebydays(outname, data_lst, label_lst, ranges_sec, p_thr, timestep):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

    nFiles = len(data_lst)
    for rng_name, rng_sec in ranges_sec.items():
        rng_min = int(rng_sec[0] / timestep)
        rng_max = int(rng_sec[1] / timestep)
        
        freqConnRange = []
        for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
            te, lag, p = data


            p_rng = p[:,:,rng_min:rng_max]
            
            freqConnRange += [np.mean(is_conn(p_rng, p_thr), axis=2)]

        N_CHANNEL = data_lst[0, 0].shape[0]
        N_CONN_EFF = N_CHANNEL * (N_CHANNEL-1)
        freqConnPerSession = [np.sum(freq) / N_CONN_EFF for freq in freqConnRange]
        avgSharedConn = [np.nan] + [np.sum(np.abs(freqConnRange[idxFile-1] - freqConnRange[idxFile])) / N_CONN_EFF for idxFile in range(1, nFiles)]

        #TODO - extrapolate
        #have_bte = "BivariateTE" in outname
        #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
        ax[0].plot(freqConnPerSession,  label=rng_name)
        ax[1].plot(avgSharedConn, '--', label=rng_name)
    
    ax[0].set_title("Average connection frequency")
    ax[1].set_title("Average connection frequency L1 distance")
    ax[0].set_xticks([])
    ax[1].set_xticks(list(range(nFiles)))
    ax[1].set_xticklabels([label[12:17] for label in label_lst])
    ax[0].legend()
    ax[1].legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_shared_link_scatter(outname, data_lst, label_lst, ranges_sec, p_thr, timestep):
    nFiles = len(data_lst)
    N_ROWS = nFiles-1
    N_COLS = len(ranges_sec)
    SHARED_MAX_EFF = 0
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(5*N_COLS, 5*N_ROWS), tight_layout=True)
    
    for iRNG, (rng_name, rng_sec) in enumerate(ranges_sec.items()):
        rng_min = int(rng_sec[0] / timestep)
        rng_max = int(rng_sec[1] / timestep)
        rng_len = rng_max - rng_min
        ax[0][iRNG].set_title(rng_name)
        
        nConnPerRng = []
        for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
            te, lag, p = data
            p_rng = p[:,:,rng_min:rng_max]
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
    
    #totalConnPerConn[-1][-1].append(np.sum(1-np.isnan(te).astype(int), axis=2).flatten()  / (te.shape[0]**2 - te.shape[0]))
    # fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    # for i, trial in enumerate(["GO","NOGO"]):#in enumerate(["ALL"]):
    #     ax1[i][0].set_ylabel(trial)
    #     for j, method in enumerate(["BTE","MTE"]):

    #         ax2[i][j].set_xlabel("connection index, sorted")
    #         ax2[i][j].set_ylabel("Frequency of occurence")

    #         thisConn = np.array(totalConnPerConn[i][j])
    #         sortedArgs = np.flip(np.argsort(np.sum(thisConn, axis=0)))

    #         for conn in totalConnPerConn[i][j]:
    #             ax2[i][j].plot(conn[sortedArgs], '.')



def plot_te_distribution(outname, data_lst, label_lst, P_THR, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data
        
        idx_conn = is_conn(p, P_THR)
        teflat = te[idx_conn.astype(bool)].flatten()
        
#         teflat = [te1 for te1,p1 in zip(te.flatten(), p.flatten()) if (not np.isnan(te1)) and (p1 < 0.01)]
        plt.hist(teflat, bins='auto', label=label[12:17], color=((idxFile / nFiles), 0, 0), alpha=0.3)

    plt.xscale("log")
    plt.xlim([1.0e-3,1])
    plt.xlabel("TE")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_rel_diff_nlinks_bytime(outname, data_lst1, data_lst2, label_lst1, label_lst2, P_THR, timestep1, timestep2):
    plt.figure(figsize=(10, 10))
    
    nFiles1 = len(data_lst1)
    nFiles2 = len(data_lst2)
    assert nFiles1 == nFiles2, "Not the same number of file types"
    
    for idxFile in range(nFiles1):
        label1 = label_lst1[idxFile]
        label2 = label_lst2[idxFile]
        assert label1 == label2, "Basenames not aligned"
        te1, lag1, p1 = data_lst1[idxFile]
        te2, lag2, p2 = data_lst2[idxFile]
        
        N_TIMES1 = te1.shape[2]
        N_TIMES2 = te2.shape[2]
        times1 = timestep1 * np.linspace(0, N_TIMES1, N_TIMES1)
        times2 = timestep2 * np.linspace(0, N_TIMES2, N_TIMES2)
        totalConnPerTime1 = np.sum(is_conn(p1, P_THR), axis=(0,1))
        totalConnPerTime2 = np.sum(is_conn(p2, P_THR), axis=(0,1))

        # If datasets have different timestep, resample to the smallest one
        RESAMPLE_PARAM = {'method' : 'averaging', 'kind' : 'kernel'}
        if N_TIMES1 > N_TIMES2:
            times1, totalConnPerTime1 = times2, resample(times1, totalConnPerTime1, times2, RESAMPLE_PARAM)
        elif N_TIMES1 < N_TIMES2:
            times2, totalConnPerTime2 = times1, resample(times2, totalConnPerTime2, times1, RESAMPLE_PARAM)

        diff = totalConnPerTime1 - totalConnPerTime2
        summ = totalConnPerTime1 + totalConnPerTime2
        
        div = np.zeros(len(times1))
        idxNonZero = summ != 0
        div[idxNonZero] = diff[idxNonZero] / summ[idxNonZero]

        plt.plot(times1, div, label=label1[12:17], color=((idxFile / nFiles1), 0, 0))

    plt.xlabel("time, seconds")
    plt.ylabel("Relative diff in number of connections")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
