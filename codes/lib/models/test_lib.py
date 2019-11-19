# Load standard libraries
import numpy as np

# Load user libraries
from codes.lib.signal_lib import approx_decay_conv, resample_kernel
from codes.lib.metrics.graph_lib import setDiagU


# Convert discretized interval to number of steps
def interv2steps(t, dt):
    return int(np.round(t / dt)) + 1

# There is one more post than intervals
def steps2interv(nt, dt):
    return (nt - 1) * dt


# Generate pure noise data
def noisePure(p):
    nT = interv2steps(p['tTot'], p['dt'])
    return np.random.normal(0, p['std'], (p['nTrial'], p['nNode'], nT))


# Generate LPF of noise data, possibly downsampled
def noiseLPF(p):
    tGrace       = p['tauConv'] * 3                        # seconds, Initial grace period to avoid accumulation effects
    nTMicroGrace = interv2steps(tGrace, p['dtMicro'])      # Number of timesteps for grace period
    nTMicroEff   = interv2steps(p['tTot'], p['dtMicro'])   # Number of timesteps for following data
    nTMicroTot   = nTMicroGrace + nTMicroEff               # Total number of timesteps

    # Micro-simulation:
    # 1) Generate random data at neuronal timescale
    # 2) Compute convolution with Ca indicator
    # 3) Downsample to experimental time-resolution, if requested
    dataRandTot = np.random.normal(0, p['std'], (nTMicroTot, p['nTrial'], p['nNode']))
    dataConvTot = approx_decay_conv(dataRandTot, p['tauConv'], p['dtMicro'])
    dataConvEff = dataConvTot[nTMicroGrace:]

    if 'dt' in p.keys():
        nT = interv2steps(p['tTot'], p['dt'])
        tArrMicro = np.linspace(0, p['tTot'], nTMicroEff)
        tArr      = np.linspace(0, p['tTot'], nT)
        downsampleKernel = resample_kernel(tArrMicro, tArr, (p['dt']/2)**2)
        #srcData = downsampleKernel.dot(dataConvEff)
        srcData = np.einsum('ij, jkl', downsampleKernel, dataConvEff) #np.matmul(downsampleKernel, dataConvEff)
    else:
        srcData = dataConvEff

    return srcData.transpose((1, 2, 0))  # [Trial x Node x Time]


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
            data[iTrial, 0, i] += param['inpMag'] * np.sin(2 * np.pi * i / param['inpT'])    # Input to the first node
            data[iTrial, :, i] += np.random.normal(0, param['std'], param['nNode'])       # Noise to all nodes

    return data


# Get true connectivity for DynSys example
def dynsys_gettrueconn(param):
    return setDiagU(param['nNode'], 1, np.nan).T
