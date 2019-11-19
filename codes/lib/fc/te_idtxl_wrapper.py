import numpy as np
import multiprocessing
import numpy as np
import pandas as pd
import pathos

# IDTxl libraries
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.multivariate_mi import MultivariateMI
from idtxl.multivariate_te import MultivariateTE
# import jpype as jp

# Local libraries
from codes.lib.aux_functions import mem_now_as_str
from codes.lib.decorators_lib import redirect_stdout, time_mem_1starg  # , jpype_sync_thread



# Convert results structure into set of matrices for better usability
# Returns shape [3 x nSource x nTarget]
def idtxlResultsParse(results, nNode, nTarget=None, method='TE', storage='matrix'):
    nNodeSrc = nNode
    nNodeTrg = nNode if nTarget is None else nTarget

    # Determine metric name to be extracted
    if 'TE' in method:
        metricName = 'selected_sources_te'
    elif 'MI' in method:
        metricName = 'selected_sources_mi'
    else:
        raise ValueError('Unexpected method', method)

    # Initialize target storage class
    if storage == 'pandas':
        cols = ['src', 'trg', 'te', 'lag', 'p']
        df = pd.DataFrame([], columns=cols)
    else:
        matTE = np.zeros((nNodeSrc, nNodeTrg)) + np.nan
        matLag = np.zeros((nNodeSrc, nNodeTrg)) + np.nan
        matP = np.zeros((nNodeSrc, nNodeTrg)) + np.nan

    # Parse data
    for iTrg in range(nNodeTrg):
        if isinstance(results, list):
            rezThis = results[iTrg].get_single_target(iTrg, fdr=False)
        else:
            rezThis = results.get_single_target(iTrg, fdr=False)

        # If any connections were found, get their data  at all was found
        if rezThis[metricName] is not None:
            te_lst = rezThis[metricName]
            p_lst = rezThis['selected_sources_pval']
            lag_lst = [val[1] for val in rezThis['selected_vars_sources']]
            src_lst = [val[0] for val in rezThis['selected_vars_sources']]
            trg_lst = [iTrg] * len(te_lst)
            rezThisZip = zip(src_lst, trg_lst, te_lst, lag_lst, p_lst)

            if storage == 'pandas':
                df = df.append(pd.DataFrame(list(rezThisZip), columns=cols), ignore_index=True)
            else:
                for iSrc, iTrg, te, lag, p in rezThisZip:
                    matTE[iSrc][iTrg] = te
                    matLag[iSrc][iTrg] = lag
                    matP[iSrc][iTrg] = p
    if storage == 'pandas':
        # df = pd.DataFrame.from_dict(out)
        df = df.sort_values(by=['src', 'trg'])
        return df
    else:
        return matTE, matLag, matP


# Construct an IDTxl analysis class using its name
def getAnalysisClass(methodname):
    # Initialise analysis object
    if   methodname == "BivariateMI":     return BivariateMI()
    elif methodname == "MultivariateMI":  return MultivariateMI()
    elif methodname == "BivariateTE":     return BivariateTE()
    elif methodname == "MultivariateTE":  return MultivariateTE()
    else:
        raise ValueError("Unexpected method", methodname)


#@jpype_sync_thread
@redirect_stdout
def parallelTask(trg, data, method, settings):
    analysisClass = getAnalysisClass(method)
    return analysisClass.analyse_single_target(settings=settings, data=data, target=trg)


# Parallelize a FC estimate over targets
def idtxlParallelCPU(data, settings, method, NCore = None):
    # Get number of processes
    idxProcesses = settings['dim_order'].index("p")
    nProcesses = data.shape[idxProcesses]
    
    # Convert data to ITDxl format
    dataIDTxl = Data(data, dim_order=settings['dim_order'])

    # Initialize multiprocessing pool
    if NCore is None:
        NCore = pathos.multiprocessing.cpu_count() - 1
    pool = pathos.multiprocessing.ProcessingPool(NCore)
    #pool = multiprocessing.Pool(NCore)

    targetLst = list(range(nProcesses))

    parallelTaskCompact = lambda trg: parallelTask(trg, dataIDTxl, method, settings)
    rez = pool.map(parallelTaskCompact, targetLst)

    return idtxlResultsParse(rez, nProcesses, method=method, storage='matrix')


# Parallelize FC estimate over targets, datasets and methods
# Returns dict {"method" : array of shape [nData x 3 x nSource x nTarget]}
def idtxlParallelCPUMulti(dataLst, settings, methods, NCore = None, serial=False, target=None):
    '''
    Performs parameter sweep over methods, data sets and channels, distributing work equally among available processes
    * Number of processes (aka channels) must be equal for all datasets
    '''

    print("Mem:", mem_now_as_str(), "- Start of subroutine")

    ##########################################
    # Determine parameters for the parameter sweep
    ##########################################
    idxProcesses = settings['dim_order'].index("p")      # Index of Processes dimension in data

    nMethods = len(methods)
    nDataSets = len(dataLst)
    nProcesses = dataLst[0].shape[idxProcesses]

    # Indices of all methods
    # Indices of all data sets
    # Indices of all target processes (aka data channels)
    mIdxs = np.arange(nMethods, dtype=int)
    dIdxs = np.arange(nDataSets, dtype=int)
    tIds  = np.arange(nProcesses, dtype=int) if target is None else np.array([target])
    nProcessesEff = len(tIds)

    sweepLst = [(m, d, t) for m in mIdxs for d in dIdxs for t in tIds]
    sweepIdxs = np.arange(len(sweepLst))

    ###############################
    # Convert data to ITDxl format
    ###############################
    dataIDTxl_lst = [Data(d, dim_order=settings['dim_order']) for d in dataLst]

    print("Mem:", mem_now_as_str(), "- Converted all data to IDTxl format")

    ###############################
    # Initialize multiprocessing pool
    ###############################
    procIdRoot = multiprocessing.current_process().pid
    if serial:
        myMap = lambda f, x: list(map(f, x))
    else:
        if NCore is None:
            NCore = pathos.multiprocessing.cpu_count() - 1
        pool = pathos.multiprocessing.ProcessingPool(NCore)
        #pool = multiprocessing.Pool(NCore)
        myMap = pool.map

    ###############################
    # Compute estimators in parallel
    ###############################
    def task_gen():
        for sw in sweepLst:
            methodIdx, dataIdx, targetIdx = sw
            yield int(targetIdx), dataIDTxl_lst[dataIdx], methods[int(methodIdx)], settings

    @time_mem_1starg
    def parallelTaskProxy(task):
        iTrg, data, method, settings = task
        return parallelTask(iTrg, data, method, settings)

    print("----Root process", procIdRoot, "started task on", NCore, "cores----")
    rez_multilst = myMap(parallelTaskProxy, task_gen())

    ###############################
    # Parse computed matrices
    ###############################
    tripleIdxs = dict(zip(sweepLst, sweepIdxs))
    rez = [[[rez_multilst[tripleIdxs[(m, d, t)]] for t in tIds] for d in dIdxs] for m in mIdxs]

    rezParsed = {
        method : np.array([idtxlResultsParse(rez[m][d], nProcessesEff, method=method, storage='matrix') for d in dIdxs])
        for m, method in enumerate(methods)
    }

    print("----Root process", procIdRoot, "finished task")
    return rezParsed
