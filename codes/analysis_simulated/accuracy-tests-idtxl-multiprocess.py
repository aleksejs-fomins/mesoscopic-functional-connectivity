from codes.lib.data_io.qt_wrapper import gui_fnames

'''
  Width/Depth Test Code:
    Load all width files
    Compute TE
    Make Plots, Store TE

  SNR Code:
    Load Typical File
    Loop over added noise / freq
    Compute TE
    Make Plots, Store TE

  Win/Lag/DS Code:
    Load Typical File
    Loop over windows, lag, ds
    Compute TE
    Make Plots, Store TE
'''

dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")

#############################
# IDTxl parameters
#############################
idtxl_settings = {
    'dim_order'       : 'ps',
    'method'          : 'MultivariateTE',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 1,
    'min_lag_sources' : 1}

idtxl_methods = ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE']

#############################
# Depth Tests
#############################

print("Performing Depth Tests")
depthFileNames = [fname for fname in dataFileNames if "depth" in fname]

for fname in depthFileNames:
    # Read file here

    for method in idtxl_methods:
        idtxl_settings['method'] = method
        te_results = np.zeros((3, N_NODE, N_NODE, N_STEP))

        for i, ndata in enumerate(ndata_lst):
            print("Processing Data", analysis, method, modelname, ndata)

            # Run calculation
            rez = idtxlParallelCPU(data_lst[i], idtxl_settings)

            # Parse Data
            te_results[..., i] = np.array(idtxlResultsParse(rez, N_NODE, method=method, storage='matrix'))

        # Plot
        fname = modelname + '_' + str(N_NODE) + method + '_' + analysis
        testplots(ndata_eff, te_results, TRUE_CONN, logx=True, percenty=True, pTHR=0.01, h5_fname=fname + '.h5',
                  fig_fname=fname + '.png')


#############################
# Width Tests
#############################



#############################
# SNR Tests
#############################

#############################
# Param Tests
#############################