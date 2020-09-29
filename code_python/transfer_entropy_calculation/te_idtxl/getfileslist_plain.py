# Standard libraries
import os,sys
import json
import numpy as np

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
codepath = os.path.join(rootpath, "code_python")
sys.path.append(codepath)
print("Appended root directory", codepath)

# User libraries
from lib.data_io.os_lib import getfiles_walk
from lib.data_io.qt_wrapper import gui_fpath

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
excl_folders=[]
#with open('excl_folders.json', 'r') as f:
#    excl_folders = json.load(f)['excl_folders']


allFoldersFiltered = []
for folder in allFolders:
  if np.sum([excl in folder for excl in excl_folders]) == 0:
     allFoldersFiltered += [folder]
  else:
     print("Excluded folder", folder)
print("Total Folders after exclusion", len(allFoldersFiltered))


# Step 2: Split files among processes. Files are sorted in priority order, because we need some results sooner than others
nMachines = int(sys.argv[1])
foldersPerMachine = {iMachine : [] for iMachine in range(nMachines)}

for iFolder, path in enumerate(allFoldersFiltered):
    foldersPerMachine[iFolder % nMachines] += [path]

for iMachine, foldersThis in foldersPerMachine.items():
    print("Folders per machine", iMachine, "are", len(foldersThis), "first one is", foldersThis[0])
    with open('foldersMachine'+str(iMachine)+'.json', 'w') as f:
        json.dump({"dataFolders" : foldersThis}, f)
