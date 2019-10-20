# Load standard libraries
import os, sys
import numpy as np

# # Add path to parent folder
# p1dir = os.path.dirname(os.path.realpath(__file__))
# pwd_lib = os.path.dirname(p1dir)
# sys.path.append(pwd_lib)

# Load user libraries
from codes.lib.signal_lib import approxDelayConv, resample
from codes.lib.metrics.graph_lib import setDiagU


# Generate pure noise data
def noisePure(p):
    NT    = int(p['tTot'] / p['dt'])
    shape = (p['nNode'], NT)
    return np.random.normal(0, p['std'], np.prod(shape)).reshape(shape)


# Generate LPF of noise data, possibly downsampled
def noiseLPF(p):
    T_SHIFT  = p['tauConv'] * 10    # seconds, Initial shift to avoid accumulation effects
    NT_SHIFT = int(T_SHIFT / p['dtMicro']) + 1
    NT_MICRO = int(p['tTot'] / p['dtMicro']) + 1
    
    if 'dt' in p.keys():
        NT = int(p['tTot'] / p['dt']) + 1
        t_arr_micro = np.linspace(0, p['tTot'], NT_MICRO)
        t_arr       = np.linspace(0, p['tTot'], NT)
        
    src_data = [[] for i in range(p['nNode'])]

    # Micro-simulation:
    # 1) Generate random data at neuronal timescale
    # 2) Compute convolution with Ca indicator
    # 3) Downsample to experimental time-resolution, if requested
    for iChannel in range(p['nNode']):
        data_rand = np.random.uniform(0, p['std'], NT_MICRO + NT_SHIFT)
        data_conv = approxDelayConv(data_rand, p['tauConv'], p['dtMicro'])
        src_data[iChannel] = data_conv[NT_SHIFT:]
        
        if 'dt' in p.keys():
            param_downsample = {'method' : 'averaging', 'kind' : 'kernel'}
            src_data[iChannel] = resample(t_arr_micro, src_data[iChannel], t_arr, param_downsample)
        
    return np.array(src_data)


# Creates a network of consecutively-connected linear first order ODE's
def dynsys(param):
    # Extract parameters
    eta = param['dt'] / param['tau']

    # Create Interaction matrix: 1st axis trg, 2nd axis src
    # Diagonal entries
    M = (1 - eta) * np.eye(param['nNode'], dtype=float)

    # Interactions
    M += eta * setDiagU(param['nNode'], 1)

    data = np.zeros((param['nNode'], param['nData']))
    for i in range(1, param['nData']):
        data[:, i] = M.dot(data[:, i-1])
        data[0, i] += param['mag'] * np.sin(2 * np.pi * i / param['inpT'])    # Input to the first node
        data[:, i] += np.random.normal(0, param['std'], param['nNode'])       # Noise to all nodes

    return data


# Get true connectivity for DynSys example
def dynsys_gettrueconn(param):
    return setDiagU(param['nNode'], 1)