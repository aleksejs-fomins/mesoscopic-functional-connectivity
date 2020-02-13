# import standard libraries
import multiprocessing
import pathos
import numpy as np
import time
import psutil

# IDTxl libraries
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

'''
  TODO: Test performance (Total time + Memory at each step) as function of
    [+] Test 1: Number of cores
    [ ] Test 2: Data size
    [+] Test 3: Data shape [pure time, almost pure trial]
    [+] Test 4: Data type: [random, lpf, AR(3)]
    [+] Test 5: Estimator (Gaussian / Kraskov)
    
    [ ] Produce reader for resulting files to plot
    [ ] (!!) Add parameter to spawn 
'''

#######################
# Generate Data
#######################

# Generate random autoregressive data of given order
def gen_ar(nData, nNode, nDepth):
    data = np.random.normal(0, 1, (nData, nNode))
    Marr = np.random.uniform(0, 1, (nDepth, nNode, nNode))

    for iTime in range(nDepth, nNode):
        for iHist in range(nDepth):
            data[iTime] += Marr[iHist].dot(data[iTime-iHist-1])
        data[iTime] /= np.linalg.norm(data[iTime])

    # Have to add some noise to data or IDTxl will blow up :D
    data += np.random.normal(0, 0.05, data.shape)
    return data


nDepth = 3     # Use AR(3) model
nNode = 12
nData = 1000

# Generate some data that has non-trivial inter-dependencies
dataDict = {
    "rand" : np.random.normal(0,1,(nData, nNode)),
    "ar"   : gen_ar(nData, nNode, nDepth)
}

#######################
# Run
#######################

def task(fname, settings, dataIDTxl, iTarget):
    timeStart = time.time()

    rez = methodClass.analyse_single_target(settings, dataIDTxl, iTarget)

    with open(fname, 'a') as file:
        procId = multiprocessing.current_process().pid
        file.write(
            "\t".join(str(k) for k in [procId, iTarget, time.time() - timeStart, psutil.virtual_memory().used]) + '\n')

    return rez


methodDict = {
    "BivariateMI" : BivariateMI,
    "BivariateTE" : BivariateTE,
    "MultivariateTE" : MultivariateTE,
}

nCoreMax = 4

for nCore in [4]: #np.flip(np.arange(1, nCoreMax+1)):
    for methodName, methodClassFunc in methodDict.items():
        for dataName, data in dataDict.items():
            for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
                for dataShape in ['trial']:
                    outname = '_'.join([dataName, dataShape, methodName, estimator, str(nCore)]) + '.txt'

                    with open(outname, 'w') as file:
                        file.write("Started :: mem = " + str(psutil.virtual_memory().used))

                    '''
                            WARNING !!!!!!!!!!!!!!!!1
                            Verify that the data passed to each run is correct, and not all doing the same stuff
                    '''

                    #######################
                    # Run IDTxl
                    #######################
                    # b) Initialise analysis object and define settings
                    methodClass = methodClassFunc()

                    settings = {'cmi_estimator': estimator,
                                'min_lag_sources': 1,
                                'max_lag_sources': 4}

                    # a) Convert data to ITDxl format
                    if dataShape == 'sample':
                        dataIDTxl = Data(data, dim_order='sp')
                    else:
                        window = settings['max_lag_sources'] + 1
                        dataIDTxl = Data(data.reshape(window, nData//window, nNode), dim_order='srp')

                    # c) Run analysis
                    task_wrapper = lambda iTarget, outname=outname, settings=settings, dataIDTxl=dataIDTxl : task(outname, settings, dataIDTxl, iTarget)

                    targetArr = list(range(nNode)) #np.arange(nNode).astype(int)
                    pool = pathos.multiprocessing.ProcessingPool(nCore)
                    pool.map(task_wrapper, targetArr)

                    with open(outname, 'a') as file:
                        file.write("Done :: mem = " + str(psutil.virtual_memory().used))
