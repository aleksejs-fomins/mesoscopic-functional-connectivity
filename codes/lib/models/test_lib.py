# Load standard libraries
import os, sys
import numpy as np

# # Add path to parent folder
# p1dir = os.path.dirname(os.path.realpath(__file__))
# pwd_lib = os.path.dirname(p1dir)
# sys.path.append(pwd_lib)

# Load user libraries
from codes.lib.signal_lib import approxDelayConv, resample_kernel
from codes.lib.metrics.graph_lib import setDiagU


# Generate pure noise data
def noisePure(p):
    nT = int(p['tTot'] / p['dt'])
    return np.random.normal(0, p['std'], (p['nTrial'], p['nNode'], nT))


# Generate LPF of noise data, possibly downsampled
def noiseLPF(p):
    tShift  = p['tauConv'] * 10    # seconds, Initial shift to avoid accumulation effects
    nTShift = int(tShift / p['dtMicro']) + 1
    nTMicro = int(p['tTot'] / p['dtMicro']) + 1
    
    if 'dt' in p.keys():
        nT = int(p['tTot'] / p['dt']) + 1
        tArrMicro = np.linspace(0, p['tTot'], nTMicro)
        tArr       = np.linspace(0, p['tTot'], nT)
        downsampleKernel = resample_kernel(tArrMicro, tArr, (p['dt']/2)**2)
        
    srcData = np.zeros((p['nTrial'], p['nNode'], nT))

    # Micro-simulation:
    # 1) Generate random data at neuronal timescale
    # 2) Compute convolution with Ca indicator
    # 3) Downsample to experimental time-resolution, if requested
    for iTrial in range(p['nTrial']):
        for iNode in range(p['nNode']):
            dataRand = np.random.uniform(0, p['std'], nTMicro + nTShift)
            dataConv = approxDelayConv(dataRand, p['tauConv'], p['dtMicro'])

            if 'dt' in p.keys():
                srcData[iTrial, iNode] = downsampleKernel.dot(dataConv[nTShift:])
            else:
                srcData[iTrial, iNode] = dataConv[nTShift:]

    return np.array(srcData)


# Creates a network of consecutively-connected linear first order ODE's
def dynsys(param):
    # Extract parameters
    eta = param['dt'] / param['tau']

    # Create Interaction matrix: 1st axis trg, 2nd axis src
    # Diagonal entries
    M = (1 - eta) * np.eye(param['nNode'], dtype=float)

    # Interactions
    M += eta * setDiagU(param['nNode'], 1)

    data = np.zeros((param['nTrial'], param['nNode'], param['nData']))
    for iTrial in range(param['nTrial']):
        for i in range(1, param['nData']):
            data[iTrial, :, i] = M.dot(data[iTrial, :, i-1])
            data[iTrial, 0, i] += param['mag'] * np.sin(2 * np.pi * i / param['inpT'])    # Input to the first node
            data[iTrial, :, i] += np.random.normal(0, param['std'], param['nNode'])       # Noise to all nodes

    return data


# Get true connectivity for DynSys example
def dynsys_gettrueconn(param):
    return setDiagU(param['nNode'], 1)