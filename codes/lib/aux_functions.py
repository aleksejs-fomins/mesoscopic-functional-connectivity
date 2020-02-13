import sys
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import bisect
import psutil


# Convert a list of string integers to a date. The integers correspond to ["YYYY", "MM", "DD"] - Others have not been tested
def strlst2date(strlst):
    return datetime(*np.array(strlst, dtype=int))


# Calculate difference in days between two dates in a pandas column
# def date_diff(l):
#     return np.array([(v - l.iloc[0]).days for v in l])
def date_diff(lst, v0):
    return np.array([(v - v0).days for v in lst])


# Get current time as string
def time_now_as_str():
    return strftime("[%Y.%m.%d %H:%M:%S]", gmtime())


# Get current memory use as string
def mem_now_as_str():
    return str(psutil.virtual_memory().used)


# Print progress bar with percentage
def progress_bar(i, imax, suffix=None):
    sys.stdout.write('\r')
    sys.stdout.write('[{:3d}%] '.format(i * 100 // imax))
    if suffix is not None:
        sys.stdout.write(suffix)
    if i == imax:
        sys.stdout.write("\n")
    sys.stdout.flush()


# Merge a list of dictionaries with exactly the same key structure
def merge_dicts(d_lst):
    return {key : [d[key] for d in d_lst] for key in d_lst[0].keys()}


# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]


# Finds permutation map A->B of elements of two arrays, which are permutations of each other
def perm_map_arr(a, b):
    return np.where(a.reshape(a.size, 1) == b)[1]


# Same as perm_map_arr, but for string characters
def perm_map_str(a, b):
    return perm_map_arr(np.array(list(a)), np.array(list(b)))


# Return original shape, but replace all axis that have been reduced with ones
# So final shape looks as if it is of the same dimension as original
# Useful for broadcasting reduced arrays onto original arrays
def reshape_reduced_axes(shapeOrig, axisReduced):
    if axisReduced is None:
        return (1,)
    else:
        if not isinstance(axisReduced, tuple):
            axisReduced = (axisReduced,)

        shapeNew = list(shapeOrig)
        for idx in axisReduced:
            shapeNew[idx] = 1
        return tuple(shapeNew)


# Assign each string to one key out of provided
# If no keys found, assign special key
# If more than 1 key found, raise error
def bin_data_by_keys(strLst, keys):
    keysArr = np.array(keys, dtype=object)
    rez = []
    for s in strLst:
        matchKeys = np.array([k in s for k in keys], dtype=bool)
        nMatch = np.sum(matchKeys)
        if nMatch == 0:
            rez += ['other']
        elif nMatch == 1:
            rez += [keysArr[matchKeys][0]]
        else:
            raise ValueError("String", s, "matched multiple keys", keysArr[matchKeys])

    assert len(rez) == len(strLst), "Resulting array length does not match original"
    return rez