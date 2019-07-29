import os
import numpy as np


# Find all folders in this folder (excluding subdirectories)
def get_subfolders(folderpath):
    return [_dir for _dir in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, _dir))]

# Find all files in a given directory including subdirectories
# All keys must appear in file name
def getfiles_walk(inputpath, keys):
    rez = []
    NKEYS = len(keys)
    for dirpath, dirnames, filenames in os.walk(inputpath):
        for filename in filenames:
            if np.sum(np.array([key in filename for key in keys], dtype=int)) == NKEYS:
                rez += [(dirpath, filename)]
    return np.array(rez)
