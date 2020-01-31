# Load standard libraries
import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

# Load user libraries
from codes.lib.data_io.matlab_lib import loadmat

'''
  TODO:
    1) Load data from HDF5
    2) Save data as temporary .mat
    3) Run matlab NIfTy to get TE
    4) Load NIfTy TE output from .mat
    5) Save results as HDF5
'''

def nifty_wrapper(src_name_h5, pwd_h5, pwd_mat):
    # 1) Load data from HDF5
    src_path_h5 = os.path.join(pwd_h5, src_name_h5)
    print("Reading source data from", src_path_h5)
    src_file_h5 = h5py.File(src_path_h5, "r")
    src_data = np.copy(src_file_h5['data'])
    src_file_h5.close()
    
    print("read data from H5 : ", src_data.shape, type(src_data))
    
    # 2) Save data as temporary .mat
    src_name_mat = 'source_' + src_name_h5.split('.')[0] + '.mat'
    rez_name_mat = 'result_' + src_name_h5.split('.')[0] + '.mat'
    src_path_mat = os.path.join(pwd_mat, src_name_mat)
    rez_path_mat = os.path.join(pwd_mat, rez_name_mat)
    print("Converting data to matlab file", src_path_mat)
    scipy.io.savemat(src_path_mat, {"data" : src_data})

    # 3) Run matlab NIfTy to get TE
    print("Started running matlab")

    action1 = 'nifty_path = "'       + os.path.join(pwd_lib, "nifty_wrapper") + '/";'
    action2 = 'source_file_name = "' + src_path_mat + '";'
    action3 = 'result_file_name = "' + rez_path_mat + '";'
    action4 = 'addpath(char(nifty_path));'
    action5 = 'run("nifty_wrapper.m");'
    action_sum = action1 + action2 + action3 + action4 + action5 + "exit;"
    
    #action1 = 'core_path = "' + path2p + '/";'
    #action2 = 'addpath(char(core_path + "codes/lib/nifty_wrapper/"));'
    #action3 = 'run("nifty_wrapper.m");'
    #action_sum = action1 + action2 + action3 + "exit;"
    
    print("..Action:", action_sum)
    subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", action_sum])
    # '"run(' +"'test_nifty_alyosha.m');exit;" + '"'

    # 4) Load NIfTy TE output from .mat
    print("Loading matlab results file", rez_path_mat)
    rez_data = loadmat(rez_path_mat)
    
    # 5) Save results as HDF5
    rez_path_h5 = os.path.join(pwd_h5, 'result_' + src_name_h5.split('.')[0] + '.h5')
    print("Writing results data to", rez_path_h5)
    
    rez_file_h5 = h5py.File(rez_path_h5, "w")
    results_grp = rez_file_h5.create_group("results")
    results_grp['data'] = rez_data['results']['data']
    results_grp['TE_table'] = rez_data['results']['TE_table']
    results_grp['p_table'] = rez_data['results']['p_table']
    results_grp['delay_table'] = rez_data['results']['delay_table']
    rez_file_h5.close()
    
    return rez_path_h5
