import bisect
import numpy as np

# Compute indices of slice of sorted data which fit into the provided range
def slice_sorted(data, rng):
    return [
        bisect.bisect_left(data, rng[0]),
        bisect.bisect_right(data, rng[1])]


# Finds permutation map A->B of elements of two arrays, which are permutations of each other
def perm_map_arr(a, b):
    return np.where(b.reshape(b.size, 1) == a)[1]


# Same as perm_map_arr, but for string characters
def perm_map_str(a, b):
    return perm_map_arr(np.array(list(a)), np.array(list(b)))


# Returns set subtraction of s1 - s2, preserving order of s1
def unique_subtract(s1, s2):
    rez = [s for s in s1 if s not in s2]
    if type(s1) == list:
        return rez
    elif type(s1) == str:
        return "".join(rez)
    elif type(s1) == tuple:
        return tuple(rez)
    else:
        raise ValueError("Unexpected Type", type(s1))


# Test if a given dimension is part of a dimension order
def test_have_dim(task, dimOrd, trgDim):
    if trgDim not in dimOrd:
        dimNameDict = {
            "p": "processes (aka channels)",
            "s": "samples (aka times)",
            "r": "repetitions (aka trials)"
        }
        raise ValueError(task, "requires", dimNameDict[trgDim], "dimension; have", dimOrd)


# Transpose data dimensions given permutation of axis labels
# If augment option is on, then extra axis of length 1 are added when missing
def numpy_transpose_byorder(data, orderSrc, orderTrg, augment=False):
    if data.ndim != len(orderSrc):
        raise ValueError("Incompatible data", data.shape, "and order", orderSrc)

    if not augment:
        if set(orderSrc) != set(orderTrg):
            raise ValueError('Cannot transform', orderSrc, "to", orderTrg)
        return data.transpose(perm_map_str(orderSrc, orderTrg))
    else:
        if not set(orderSrc).issubset(set(orderTrg)):
            raise ValueError('Cannot augment', orderSrc, "to", orderTrg)
        nIncr = len(orderTrg) - len(orderSrc)
        newShape = data.shape + tuple([1]*nIncr)
        newOrder = orderSrc + unique_subtract(orderTrg, orderSrc)

        return data.reshape(newShape).transpose(perm_map_str(newOrder, orderTrg))


# Return original shape, but replace all axis that have been reduced with ones
# So final shape looks as if it is of the same dimension as original
# Useful for broadcasting reduced arrays onto original arrays
def numpy_shape_reduced_axes(shapeOrig, reducedAxis):
    if reducedAxis is None:  # All axes have been reduced
        return tuple([1]*len(shapeOrig))
    else:
        if not isinstance(reducedAxis, tuple):
            reducedAxis = (reducedAxis,)

        shapeNew = list(shapeOrig)
        for idx in reducedAxis:
            shapeNew[idx] = 1
        return tuple(shapeNew)


# Add extra dimensions of size 1 to array at given locations
def numpy_add_empty_axes(x, axes):
    newShape = list(x.shape)
    for axis in axes:
        newShape.insert(axis, 1)
    return x.reshape(tuple(newShape))


# Reshape array by merging all dimensions between l and r
def numpy_merge_dimensions(data, l, r):
    shOrig = list(data.shape)
    shNew = tuple(shOrig[:l] + [np.prod(shOrig[l:r])] + shOrig[r:])
    return data.reshape(shNew)


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