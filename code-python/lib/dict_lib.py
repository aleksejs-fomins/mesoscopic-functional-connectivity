# Merge a list of dictionaries with exactly the same key structure
def merge_dicts(d_lst):
    return {key : [d[key] for d in d_lst] for key in d_lst[0].keys()}