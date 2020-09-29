from lib.stat.stat_lib import rand_bool_perm

def split_train_test(x, y, fracTest):
    n = len(x)
    nTest = int(fracTest * n)
    testIdxs = rand_bool_perm(nTest, n)
    xTrain, yTrain = x[~testIdxs], y[~testIdxs]
    xTest, yTest = x[testIdxs], y[testIdxs]
    return xTrain, yTrain, xTest, yTest