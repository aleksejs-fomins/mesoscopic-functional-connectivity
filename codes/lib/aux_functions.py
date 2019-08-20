import sys

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

