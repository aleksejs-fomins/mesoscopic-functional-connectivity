# Standard libraries
import os,sys
import json
import numpy as np

# Append base directory
currentdir = os.path.dirname(os.path.abspath(__file__))
path1p = os.path.dirname(currentdir)
path2p = os.path.dirname(path1p)
libpath = os.path.join(path2p, "lib")

sys.path.insert(0, libpath)
print("Appended library directory", libpath)

# User libraries
from data_io.os_lib import getfiles_walk
from data_io.qt_wrapper import gui_fpath

# Step 1: Get all data files
thisFolder = './'
allFolders = []
while thisFolder != '':
    thisFolder = gui_fpath("Root folder for data files", thisFolder)
    if thisFolder != '':
        folders = getfiles_walk(thisFolder, ['behaviorvar'])[:, 0]
        allFolders += list(folders)

print("Total Folders Found", len(allFolders))


# Exclude some data that is already done or bad
excl_folders = [
 'mvg_7_2018_11_09_a',
 'mvg_7_2018_11_14_a',
 'mvg_7_2018_11_15_a',
 'mvg_7_2018_11_17_a',
 'mvg_7_2018_11_18_a',
 'mvg_7_2018_11_23_a',
 'mvg_7_2018_11_26_a',
 'mvg_7_2018_12_04_a']

allFoldersFiltered = []
for folder in allFolders:
  if np.sum([excl in folder for excl in excl_folders]) == 0:
     allFoldersFiltered += [folder]
  else:
     print("Excluded folder", folder)
print("Total Folders after exclusion", len(allFoldersFiltered))


# Step 2: Split files among processes. Files are sorted in priority order, because we need some results sooner than others
nMachines = 20
foldersPerMachine = {iMachine : [] for iMachine in range(nMachines)}

for iFolder, path in enumerate(allFoldersFiltered):
    foldersPerMachine[iFolder % nMachines] += [path]

for iMachine, foldersThis in foldersPerMachine.items():
    print("Folders per machine", iMachine, "are", len(foldersThis), "first one is", foldersThis[0])
    with open('foldersMachine'+str(iMachine)+'.json', 'w') as f:
        json.dump({"dataFolders" : foldersThis}, f)
