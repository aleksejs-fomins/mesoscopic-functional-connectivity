import sys
import bisect

# Print progress bar with percentage
def progress_bar(i, imax, suffix=None):
    sys.stdout.write('\r')
    sys.stdout.write('[{:3d}%] '.format(i * 100 // imax))
    if suffix is not None:
        sys.stdout.write(suffix)
    if i == imax:
        sys.stdout.write("\n")
    sys.stdout.flush()

# Merge 2 dictionaries, given that values of both are lists
def merge_dicts(d_lst):
    d_rez = d_lst[0]
    for i in range(1, len(d_lst)):
        d_rez = {k1 : v1 + d_lst[i][k1] for k1, v1 in d_rez.items()}
    return d_rez

# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]