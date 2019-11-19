import os, sys
from os.path import basename, dirname, join, abspath
import datetime
import numpy as np
import pandas as pd

from codes.lib.data_io.os_lib import getfiles_walk

def mousekey2date(key):
    return datetime.datetime(*np.array(key.split("_")[2:5], dtype=int))

def parseFoldersMulti(root_path_data, root_path_paw, root_path_lick, root_path_whisk):
    pathDataSets = {}
    
    ##################################
    # Get channel labels for all mice
    ##################################
    print("Searching for channel labels")
    labels_paths = getfiles_walk(root_path_data, ['channel_labels.mat'])
    channel_data = [[basename(path), join(path, name)] for path, name in labels_paths]
    channel_dict = {k : v for k,v in zip(['mousename', 'path'], np.array(channel_data).T) }
    pathDataSets['channel_labels'] = pd.DataFrame(channel_dict)

    ##################################
    # Get data folders for all days
    ##################################
    print("Searching for data files")
    data_paths = getfiles_walk(root_path_data, ["data.mat"])
    neuro_data = [[basename(path), path, basename(dirname(path)), mousekey2date(basename(path))] for path, name in data_paths]
    neuro_dict = {k : v for k,v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(neuro_data).T) }
    pathDataSets['data_path'] = pd.DataFrame(neuro_dict)

    ##################################
    # Get paw folders for all mice
    ##################################
    print("Searching for paw files")
    paw_paths = getfiles_walk(root_path_paw, ["deltaI_paw.mat"])
    paw_data = [[basename(path), path, basename(dirname(path)), mousekey2date(basename(path))] for path, name in paw_paths]
    paw_dict = {k : v for k,v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(paw_data).T) }
    pathDataSets['paw_path'] = pd.DataFrame(paw_dict)

    ##################################
    # Get lick folders for all mice
    ##################################
    print("Searching for lick files")
    lick_paths = getfiles_walk(root_path_lick, ["lick_traces.mat"])
    lick_data = [[basename(path), path, basename(dirname(path)), mousekey2date(basename(path))] for path, name in lick_paths]
    lick_dict = {k : v for k,v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(lick_data).T) }
    pathDataSets['lick_path'] = pd.DataFrame(lick_dict)

    ##################################
    # Get whisk folders for all mice
    ##################################
    print("Searching for whisk files")
    whisk_paths = getfiles_walk(root_path_whisk, ["whiskAngle.mat"])
    whisk_data = [[basename(path), path, basename(dirname(path)), mousekey2date(basename(path))] for path, name in whisk_paths]
    whisk_dict = {k : v for k,v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(whisk_data).T) }
    pathDataSets['whisk_path'] = pd.DataFrame(whisk_dict)

    ##################################
    # Print results
    ##################################
    mice = list(set(pathDataSets['channel_labels']['mousename']))
    sumByMouse = lambda dataset : [dataset[dataset['mousename'] == mousename].shape[0] for mousename in mice]

    summary = pd.DataFrame({
        "Data"  : sumByMouse(pathDataSets['data_path']),
        "Paw"   : sumByMouse(pathDataSets['paw_path']),
        "Lick"  : sumByMouse(pathDataSets['lick_path']),
        "Whisk" : sumByMouse(pathDataSets['whisk_path'])
    }, index=mice)
    
    return pathDataSets, summary