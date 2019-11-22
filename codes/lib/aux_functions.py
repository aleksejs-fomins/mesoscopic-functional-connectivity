import sys
import numpy as np
from time import gmtime, strftime
import bisect
import psutil

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

def perm_map(A, B):
    aArr = np.array(A)
    bArr = np.array(B)
    return np.where(aArr.reshape(aArr.size, 1) == bArr)[1]