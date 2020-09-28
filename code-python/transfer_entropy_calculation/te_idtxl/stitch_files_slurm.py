# Standard libraries
import os,sys
import h5py
import pandas as pd
import numpy as np

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# User libraries
from codes.lib.data_io.qt_wrapper import gui_fnames, gui_fpath

'''
1. Parse all file names into Pandas
2. For every (mouse, trialType, method) create file, pull all rows
3. Guess sweepRange and nNode from set of those values
4. Construct matrix based on extent
5. Loop over all rows, read files, fill matrix, save 
'''

###################
#  Params
###################

maxlag = 3

###################
#  Get source files and target directory
###################

idtxlRezFilePathNames = gui_fnames("Open IDTxl result files to stitch", "./", "HDF5 (*.h5)")
resultPath = gui_fpath("Path to save results", "./")

###################
#  Parse parameter values from file names
###################

rows = [os.path.splitext(os.path.basename(fname))[0].split('_') for fname in idtxlRezFilePathNames]
df = pd.DataFrame(rows, columns=("fname", "swipe", "trialType", "sweep", "method", "target"))
df.drop(columns=["swipe"])
df = df.astype({"sweep" : "int", "target" : "int"})

srcNames   = set(df["fname"])
trialTypes = set(df["trialType"])
methods    = set(df["method"])


###################
#  Iterate over parameter combinations for different save files
###################

for srcName in srcNames:
    for trialType in trialTypes:
        for method in methods:

            ###################
            #  Extract all files that need to be stitched into 1
            ###################

            rows = df[
                (df["fname"]==srcName)&
                (df["trialType"] == trialType) &
                (df["method"] == method)
            ]

            sweeps = set(rows["sweep"])
            targets = set(rows["targets"])

            nTimes = np.max(sweeps) + maxlag + 1
            nNode = np.max(targets) + 1

            ###################
            #  Read and stitch those files
            ###################

            resultMat = np.full((3, nNode, nNode, nTimes), np.nan)

            for rowIdx, row in rows.iterrows():
                fpathnameThis = idtxlRezFilePathNames[rowIdx]
                iTrg = row["target"]
                iSweep = row["sweep"] + maxlag

                with h5py.File(fpathnameThis, "r") as h5f_in:
                    resultMat[:, :, iTrg, iSweep] = np.copy(h5f_in["data"])

            ###################
            #  Save stitched results to new file
            ###################

            outNameBare = "_".join([srcName, trialType, method, "swipe"])
            savename = os.path.join(resultPath, outNameBare + '.h5')
            with h5py.File(savename, "w") as h5f_out:
                grp_rez = h5f_out.create_group("results")
                grp_rez['TE_table']    = resultMat[0]
                grp_rez['delay_table'] = resultMat[1]
                grp_rez['p_table']     = resultMat[2]
