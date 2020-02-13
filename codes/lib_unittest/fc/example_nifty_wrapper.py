# Load standard libraries
import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

# Load user libraries
from codes.lib.signal_lib import approx_decay_conv
from codes.lib.info_metrics.te_nifty_wrapper.nifty_wrapper import nifty_wrapper

########################
## Generate input data
########################

'''
1) Generate random data of 10s with step 1ms
2) Apply DecayConv with tau=0.5
3) Bin using 200ms bins
'''

# #y = 1j*np.random.normal(0, 1, 10000)
# #y += np.random.normal(0, 1, 10000)
# y = 1j*np.linspace(1, 0, 10000)
# y += np.linspace(1, 0, 10000)


# yf = np.abs(np.fft.ifft(y))
# # Also normalize it
# yf /= np.max(yf)

# plt.figure()
# plt.plot(yf)
# plt.show()


print("Generating data")

N_CHANNEL = 12
N_TRIAL = 200
T_TOT = 10                  # seconds, Total simulation time
TAU_CONV = 0.5              # seconds, Ca indicator decay constant
T_SHIFT  = TAU_CONV * 10    # seconds, Initial shift to avoid accumulation effects
DT_MICRO = 0.001            # seconds, Neuronal spike timing resolution
DT_MACRO = 0.2              # seconds, Binned optical recording resolution
NT_MICRO       = int(T_TOT / DT_MICRO)
NT_MICRO_SHIFT = int(T_SHIFT / DT_MICRO)
NT_MACRO       = int(T_TOT / DT_MACRO)

dimRez = (N_TRIAL, NT_MACRO, N_CHANNEL)
dimRezMult = dimRez[0]*dimRez[1]*dimRez[2]
#src_data = np.random.uniform(0, 1, dimRezMult).reshape(dimRez)
src_data = np.zeros(dimRez)

for iTrial in range(N_TRIAL):
    for iChannel in range(N_CHANNEL):
        # Micro-simulation:
        # 1) Generate random data at neuronal timescale
        # 2) Compute convolution with Ca indicator
        data_micro = np.random.uniform(0, 1, NT_MICRO + NT_MICRO_SHIFT)
        data_micro_conv = approx_decay_conv(data_micro, TAU_CONV, DT_MICRO)[NT_MICRO_SHIFT:]
            
        # Downsampling:
        # * Bin convolved data to sampling timescale
        src_data[iTrial, :, iChannel] = np.mean(data_micro_conv.reshape((NT_MACRO, NT_MICRO//NT_MACRO)), axis=1)
    
#plt.figure()
#plt.plot(np.linspace(0, T_TOT, NT_MICRO), data_micro_conv)
#plt.plot(np.linspace(0, T_TOT, NT_MACRO), data_macro)
#plt.show()

#plt.figure()
#plt.plot(src_data[0,:,0])
#plt.plot(src_data[0,:,1])
#plt.plot(src_data[0,:,2])
#plt.show()


########################
## Save result as HDF5
########################
src_path_h5 = os.path.join(pwd_rez, "source_selftest_rand.h5")
print("Writing source data to", src_path_h5)
src_file_h5 = h5py.File(src_path_h5, "w")
src_file_h5['data'] = src_data
src_file_h5.close()

#######################
# Run NIFTY wrapper
#######################
rez_path_h5 = nifty_wrapper(src_path_h5, pwd_rez)

#######################
# Load NIFTY result from HDF5
#######################
rez_file_h5 = h5py.File(rez_path_h5, "r")
rez_data = rez_file_h5['results']

######################
# Analysis
#
# Compute mean and var TE
# Compute mean and var P-val
# Compute num of P-val < 0.01
# Compute mean and var TE for P-val < 0.01
######################

print("Plotting results")
dim = rez_data['TE_table'].shape

print(dim)

#fig, ax = plt.subplots(ncols = 2)
#ax[0].set_title('TE')
#ax[1].set_title('P-val')
#for i in range(dim[0]):
    #for j in range(dim[1]):
        #if i != j:
            #ax[0].plot(rez_data['TE_table'][i][j])
            #ax[1].plot(rez_data['p_table'][i][j])
            
#plt.show()

te_1D = []
p_1D = []
for i in range(dim[0]):
    for j in range(dim[1]):
        if i != j:
            te_1D += list(rez_data['TE_table'][i][j])
            p_1D += list(rez_data['p_table'][i][j])

p_thr = 0.01
te_1D = np.array([te for te in te_1D if not np.isnan(te)])
p_1D  = np.array([p for p in p_1D if not np.isnan(p)])
te_1D_filter = te_1D[p_1D < p_thr]

print("Freq of p-val <", p_thr, "is", np.sum(p_1D < p_thr) / len(p_1D))

fig, ax = plt.subplots(ncols=2, nrows=2)
ax[0][0].set_title("TE")
ax[0][1].set_title("P")
ax[1][0].set_title("TE @ P < " + str(p_thr))
ax[0][0].hist(te_1D)
ax[0][1].hist(p_1D)
ax[1][0].hist(te_1D_filter)
plt.show()

rez_file_h5.close()
