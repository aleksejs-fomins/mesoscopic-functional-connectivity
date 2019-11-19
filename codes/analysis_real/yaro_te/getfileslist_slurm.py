'''
  Plan:
  1. Get all files:
  2. For each file, on 1 core do 1 target, all sweeps
'''


# Standard libraries
import os,sys
import json
import numpy as np

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# User libraries
from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.qt_wrapper import gui_fpath

# Step 1: Get all data files
thisFolder = './'
allFolders = []
while thisFolder != '':
    thisFolder = gui_fpath("Root folder for data files", thisFolder)
    if thisFolder != '':
        folders = getfiles_walk(thisFolder, ['behaviorvar'])[:, 0]
        allFolders += list(folders)

print("Total Folders Found", len(allFolders))


# # Exclude some data that is already done or bad
# with open('excl_folders.json', 'r') as f:
#    excl_folders = json.load(f)['excl_folders']
#
# allFoldersFiltered = []
# for folder in allFolders:
#   if not np.any([excl in folder for excl in excl_folders]):
#      allFoldersFiltered += [folder]
#   else:
#      print("Excluded folder", folder)
# print("Total Folders after exclusion", len(allFoldersFiltered))
# allFolders = allFoldersFiltered


# Step 2: Split files among processes. Files are sorted in priority order, because we need some results sooner than others

taskIdx2task = []

for fpath in allFolders:
    nChannlel = 48 if "mvg_48" in fpath else 12

    for iTrg in range(nChannlel):
        taskIdx2task += [(fpath, iTrg)]

outfname = 'slurmtasks.json'
print("Writing", len(taskIdx2task), "tasks to", outfname)
with open(outfname, 'w') as f:
    json.dump(taskIdx2task, f, indent=4, sort_keys=True)
