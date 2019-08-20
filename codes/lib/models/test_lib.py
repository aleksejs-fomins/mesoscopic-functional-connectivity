# Load standard libraries
import os, sys
import numpy as np

# Add path to parent folder
p1dir = os.path.dirname(os.path.realpath(__file__))
pwd_lib = os.path.dirname(p1dir)
sys.path.append(pwd_lib)

# Load user libraries
from signal_lib import approxDelayConv, resample


# Generate pure noise data
def noisePure(p):
    NT    = int(p['T_TOT'] / p['DT'])
    shape = (p['N_NODE'], NT)
    return np.random.normal(0, p['STD'], np.prod(shape)).reshape(shape)
    

# Generate LPF of noise data, possibly downsampled
def noiseLPF(p):
    T_SHIFT  = p['TAU_CONV'] * 10    # seconds, Initial shift to avoid accumulation effects
    NT_SHIFT = int(T_SHIFT / p['DT_MICRO']) + 1
    NT_MICRO = int(p['T_TOT'] / p['DT_MICRO']) + 1
    
    if 'DT' in p.keys():
        NT = int(p['T_TOT'] / p['DT']) + 1
        t_arr_micro = np.linspace(0, p['T_TOT'], NT_MICRO)
        t_arr       = np.linspace(0, p['T_TOT'], NT)
        
    src_data = [[] for i in range(p['N_NODE'])]

    # Micro-simulation:
    # 1) Generate random data at neuronal timescale
    # 2) Compute convolution with Ca indicator
    # 3) Downsample to experimental time-resolution, if requested
    for iChannel in range(p['N_NODE']):
        data_rand = np.random.uniform(0, p['STD'], NT_MICRO + NT_SHIFT)
        data_conv = approxDelayConv(data_rand, p['TAU_CONV'], p['DT_MICRO'])
        src_data[iChannel] = data_conv[NT_SHIFT:]
        
        if 'DT' in p.keys():
            param_downsample = {'method' : 'averaging', 'kind' : 'kernel'}
            src_data[iChannel] = resample(t_arr_micro, src_data[iChannel], t_arr, param_downsample)
        
    return np.array(src_data)


# Sample short trials from one long trial
def sampleTrials(data2D, N_TRIAL, N_DATA_TRIAL):
    N_NODE, N_DATA = data2D.shape
    shape3D = (N_TRIAL, N_DATA_TRIAL, N_NODE)
    data3D = np.zeros(shape3D)
    
    startTimes = np.random.randint(0, N_DATA - N_DATA_TRIAL, N_TRIAL)
    for iTrial, t in enumerate(startTimes):
        data3D[iTrial, :, :] = data2D[:, t:t+N_DATA_TRIAL].transpose()
            
    return data3D
