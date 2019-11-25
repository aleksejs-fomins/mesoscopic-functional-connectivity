import numpy as np
import multiprocessing, pathos

from codes.lib.aux_functions import mem_now_as_str
from codes.lib.decorators_lib import time_mem_1starg
import codes.lib.fc.idtxl_wrapper as idtxl_wrapper
from codes.lib.fc.corr_lib import crossCorr


'''
Keywords
  * network         - analysis is performed on all pairs of sources and targets, returning square matrix
  * single_target   - analysis is performed on only one target, returning a vector for that target
  * parallel        - parallelize using pathos multiprocessing library
  * multiparam      - sweep over a list of datasets and methods, parallelize over all
    * target        - also sweep and parallelize over each target
    * network       - same as target, but all targets are analysed simultaneously by one local method.
    
Note that some methods do not have a single target implementation, so can't parallelize over targets, only over data and methods.
'''

# A class that switches between serial and parallel mappers
# Also deletes parallel mapper when class is deleted
class genericMapper():
    def __init__(self, serial, nCore=None):
        self.serial = serial
        self.pid = multiprocessing.current_process().pid

        if serial:
            self.nCore = 1
        else:
            self.nCore = nCore if nCore is not None else pathos.multiprocessing.cpu_count() - 1
            self.pool = pathos.multiprocessing.ProcessingPool(self.nCore)
            # self.pool = multiprocessing.Pool(self.nCore)

    # def __del__(self):
    #     if not self.serial:
    #         self.pool.close()
    #         self.pool.join()

    def map(self, f, x):
        print("----Root process", self.pid, "started task on", self.nCore, "cores----")
        rez = list(map(f, x)) if self.serial else self.pool.map(f, x)
        print("----Root process", self.pid, "finished task")
        return rez


def preprocess_data(data, library, settings):
    if library == 'idtxl':
        return idtxl_wrapper.preprocess_data(data, settings)
    else:
        return data


#@time_mem_1starg
def analyse_single_target(iTrg, data, library, method, settings):
    if library == 'idtxl':
        return idtxl_wrapper.analyse_single_target(iTrg, data, method, settings)
    else:
        raise ValueError("Single target not implemented for", library)


#@time_mem_1starg
def analyse_network(data, library, method, settings):
    if library == 'idtxl':
        return idtxl_wrapper.analyse_network(data, method, settings)
    elif library == 'corr':
        return crossCorr(data, settings, est=method)
    else:
        raise ValueError("Unexpected library", library)


def fc_single_target(iTrg, data, library, method, settings):
    # Preprocess data if necessary
    dataPreprocessed = preprocess_data(data, library, settings)

    # Compute single target FC for preprocessed data
    return analyse_single_target(iTrg, dataPreprocessed, library, method, settings)


# Parallelize a FC estimate over targets
def fc_parallel_target(data, library, method, settings, serial=False, nCore=None):
    # Get number of nodes
    idxNodeShape = settings['dim_order'].index("p")
    nNode = data.shape[idxNodeShape]

    # Preprocess data if necessary
    dataPreprocessed = preprocess_data(data, library, settings)

    # Initialize mapper depending on whether we do parallel or serial computation
    mapper = genericMapper(serial, nCore=nCore)

    # Construct task generator
    def task_gen():
        for iTrg in range(nNode):
            yield iTrg, dataPreprocessed, library, method, settings

    # Ugly intermediate function to unpack tuple
    parallel_task_proxy = lambda task : analyse_single_target(*task)

    rez = mapper.map(parallel_task_proxy, task_gen())
    return np.array(rez).transpose((1,2,0))  # (nTrg, 3, nSrc) -> (3, nSrc, nTrg)


# Switch between serial analysis of whole network simultaneously, or parallel analysis of one target at a time
def fc_parallel(data, library, method, settings, parTarget=True, serial=False, nCore=None):
    if parTarget:
        return fc_parallel_target(data, library, method, settings, serial=serial, nCore=nCore)
    else:
        return analyse_network(data, library, method, settings)


# Parallelize FC estimate over targets, datasets and methods
# Returns dict {"method" : array of shape [nData x 3 x nSource x nTarget]}
def fc_parallel_multiparam_target(dataLst, library, methods, settings, serial=False, targets=None, nCore=None):
    '''
    Performs parameter sweep over methods, data sets and channels, distributing work equally among available processes
    * Number of processes (aka channels) must be equal for all datasets
    '''

    print("Mem:", mem_now_as_str(), "- Start of subroutine")

    ##########################################
    # Determine parameters for the parameter sweep
    ##########################################
    idxProcesses = settings['dim_order'].index("p")  # Index of Processes dimension in data
    nProcesses = dataLst[0].shape[idxProcesses]

    nMethods = len(methods)
    nDataSets = len(dataLst)

    # Indices of all methods
    # Indices of all data sets
    # Indices of all target processes (aka data channels)
    mIdxs = np.arange(nMethods, dtype=int)
    dIdxs = np.arange(nDataSets, dtype=int)
    tIds = np.arange(nProcesses, dtype=int) if targets is None else targets

    sweepLst = [(m, d, t) for m in mIdxs for d in dIdxs for t in tIds]
    tripleIdxs = dict(zip(sweepLst, np.arange(len(sweepLst))))

    ###############################
    # Preprocess data
    ###############################
    dataPreprocessedLst = [preprocess_data(data, library, settings) for data in dataLst]
    print("Mem:", mem_now_as_str(), "- Converted all data to IDTxl format")

    ###############################
    # Initialize mapper
    ###############################
    mapper = genericMapper(serial, nCore=nCore)

    ###############################
    # Compute estimators in parallel
    ###############################
    def task_gen():
        for sw in sweepLst:
            methodIdx, dataIdx, targetIdx = sw
            yield int(targetIdx), dataPreprocessedLst[dataIdx], library, methods[int(methodIdx)], settings

    # Ugly intermediate function to unpack tuple
    parallel_task_proxy = lambda task : analyse_single_target(*task)

    rez_multilst = mapper.map(parallel_task_proxy, task_gen())

    ###############################
    # Glue computed matrices
    ###############################
    # Convert data [nMethod, nData, nTrg, 3, nSrc] -> {method : [nData, 3, nSrc, nTrg]}
    return {
        method: np.array([[rez_multilst[tripleIdxs[(m, d, t)]] for t in tIds] for d in dIdxs]).transpose((0, 2, 3, 1))
        for m, method in enumerate(methods)
    }


# Parallelize FC estimate over datasets and methods
# All targets are computed simultaneously on one process
# Returns dict {"method" : array of shape [nData x 3 x nSource x nTarget]}
def fc_parallel_multiparam_network(dataLst, library, methods, settings, serial=False, nCore=None):
    print("Mem:", mem_now_as_str(), "- Start of subroutine")

    ##########################################
    # Determine parameters for the parameter sweep
    ##########################################
    nMethods = len(methods)
    nDataSets = len(dataLst)

    # Indices of all methods
    # Indices of all data sets
    # Indices of all target processes (aka data channels)
    mIdxs = np.arange(nMethods, dtype=int)
    dIdxs = np.arange(nDataSets, dtype=int)

    sweepLst = [(m, d) for m in mIdxs for d in dIdxs]
    doubleIdxs = dict(zip(sweepLst, np.arange(len(sweepLst))))

    ###############################
    # Preprocess data
    ###############################
    dataPreprocessedLst = [preprocess_data(data, library, settings) for data in dataLst]
    print("Mem:", mem_now_as_str(), "- Converted all data to IDTxl format")

    ###############################
    # Initialize mapper
    ###############################
    mapper = genericMapper(serial, nCore=nCore)

    ###############################
    # Compute estimators in parallel
    ###############################
    def task_gen():
        for sw in sweepLst:
            methodIdx, dataIdx = sw
            yield dataPreprocessedLst[dataIdx], library, methods[int(methodIdx)], settings

    # Ugly intermediate function to unpack tuple
    parallel_task_proxy = lambda task : analyse_network(*task)

    rez_multilst = mapper.map(parallel_task_proxy, task_gen())

    ###############################
    # Glue computed matrices
    ###############################
    # Convert data [nMethod, nData, 3, nSrc, nTrg] -> {method : [nData, 3, nSrc, nTrg]} - no transposes needed
    return {
        method: np.array([rez_multilst[doubleIdxs[(m, d)]] for d in dIdxs])
        for m, method in enumerate(methods)
    }


# User can provide parameter whether to force-analyse each target separately or all simultaneously
def fc_parallel_multiparam(dataLst, library, methods, settings, parTarget=True, serial=False, nCore=None):
    if parTarget:
        return fc_parallel_multiparam_target(dataLst, library, methods, settings, serial=serial, nCore=nCore)
    else:
        return fc_parallel_multiparam_network(dataLst, library, methods, settings, serial=serial, nCore=nCore)