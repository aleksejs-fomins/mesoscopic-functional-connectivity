import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from signal_lib import resample


def plot_te_nconn_bytime(outname, data_lst, label_lst, timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data

        times = timestep * np.linspace(0, te.shape[2], te.shape[2])
        totalConnPerTime = [np.sum(1-np.isnan(te[:,:,i]).astype(int)) for i in range(te.shape[2])]
        
        plt.plot(times, totalConnPerTime, label=label[12:17], color=((idxFile / nFiles), 0, 0))

    have_bte = "BivariateTE" in outname
    plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
    plt.xlabel("time, seconds")
    plt.ylabel("Number of connections")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_nconn_rangebydays(outname, data_lst, label_lst, ranges_step, timestep):
    plt.figure(figsize=(10, 10))

    
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data
        te_rng = te[:,:,ranges_step[0]:ranges_step[1]]
        atLeast1ConnPerSession += [(np.sum(1-np.isnan(te_rng).astype(int), axis=2) > 0).astype(int)]
        
    totalConnPerSession = [np.sum(c) for c in atLeast1ConnPerSession]
    sharedConn = [np.nan] + [np.sum(atLeast1ConnPerSession[idxFile-1] + atLeast1ConnPerSession[idxFile] == 2) for idxFile in range(1, nFiles)]

    #TODO - extrapolate
    #have_bte = "BivariateTE" in outname
    #plt.axhline(y=4.0 if have_bte else 1.0, linestyle="--", label='chance')
    plt.plot(totalConnPerSession, label='total')
    plt.plot(sharedConn, '--',    label='shared')
    
    plt.xticks(list(range(np.sum(idxs_ths))))
    plt.xticklabels([label[12:17] for label in label_lst])
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_shared_link_scatter(outname, data_lst, label_lst, ranges_step, timestep):
    # Compute number of connections per range
    nConnPerRng = []
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data
        te_rng = te[:,:,ranges_step[0]:ranges_step[1]]
        nConnPerRng += [np.sum(1-np.isnan(te_rng).astype(int), axis=2)]

    nFiles = len(data_lst)
    rng_len = ranges_step[1] - ranges_step[0] + 1
    
    # Compute and plot number of shared connections
    sharedConn = []
    fig, ax = plt.subplots(nrows = nFiles-1, figsize=(10*(nFiles-1), 10))
    for idxFile in range(1, nFiles):
        M = np.zeros((rng_len, rng_len))
        for x, y in zip(nConnPerRng[idxFile-1].flatten(), nConnPerRng[idxFile].flatten()):
            M[x][y] += 1

        ax[idxFile-1].imshow(M)
        ax[idxFile-1].set_xlabel(label[idxFile-1][12:17])
        ax[idxFile-1].set_ylabel(label[idxFile][12:17])
        
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



def plot_te_distribution(outname, data_lst, label_lst,timestep):
    plt.figure(figsize=(10, 10))
    
    nFiles = len(data_lst)
    for idxFile, (data, label) in enumerate(zip(data_lst, label_lst)):
        te, lag, p = data
        
        teflat = [te1 for te1,p1 in zip(te.flatten(), p.flatten()) if (not np.isnan(te1)) and (p1 < 0.01)]
        plt.hist(teflat, bins='auto', label=label[12:17], color=((idxFile / nFiles), 0, 0), alpha=0.3)

    plt.xlabel("TE")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(outname)
    plt.close()
    
    
def plot_te_rel_diff_nlinks_bytime(outname, data_lst1, data_lst2, label_lst1, label_lst2, timestep1, timestep2):
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
        
        times1 = timestep1 * np.linspace(0, te1.shape[2], te1.shape[2])
        times2 = timestep2 * np.linspace(0, te2.shape[2], te2.shape[2])
        totalconn = lambda te : np.array([np.sum(1-np.isnan(te[:,:,i]).astype(int)) for i in range(te.shape[2])])
        totalConnPerTime1 = totalconn(te1)
        totalConnPerTime2 = totalconn(te2)

        # If datasets have different timestep, resample to the smallest one
        RESAMPLE_PARAM = {'method' : 'averaging', 'kind' : 'kernel'}
        if te1.shape[2] > te2.shape[2]:
            times1, totalConnPerTime1 = times2, resample(times1, totalConnPerTime1, times2, RESAMPLE_PARAM)
        elif te1.shape[2] < te2.shape[2]:
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
    
    
