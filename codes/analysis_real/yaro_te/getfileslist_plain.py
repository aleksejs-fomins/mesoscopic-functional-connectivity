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

# Step 2: Split files among processes. Files are sorted in priority order, because we need some results sooner than others
nMachines = 10
foldersPerMachine = [[]]*nMachines

for iFolder, path in enumerate(allFolders):
    foldersPerMachine[iFolder % nMachines] += [path]
    
for iMachine, foldersThis in enumerate(foldersPerMachine):
    with open('foldersMachine'+str(i)+'.json', 'w') as f:
        json.dump({"dataFolders" : foldersThis}, f)