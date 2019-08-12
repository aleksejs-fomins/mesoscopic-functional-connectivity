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
    
    
def plot_te_distribution(outname, data_lst, label_lst, timestep):
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
    
    
