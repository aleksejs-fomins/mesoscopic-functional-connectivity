import sys
import numpy as np
from time import gmtime, strftime
from datetime import datetime
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
