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


# Merge a list of dictionaries with exactly the same key structure
def merge_dicts(d_lst):
    return {key : [d[key] for d in d_lst] for key in d_lst[0].keys()}


# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]