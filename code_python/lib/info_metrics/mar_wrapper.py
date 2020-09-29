import numpy as np

from lib.signal_lib import zscore
from lib.array_lib import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from lib.info_metrics.mar.autoregression1D import AR1D
from lib.stat.ml_lib import split_train_test


# Compute the coefficient of the AR(1) process
def ar1_coeff(data, settings, testFrac=0.1):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 1, 3)
    dataThis = zscore(dataFlat)

    ar1D = AR1D(nHist=1)
    x, y = ar1D.data2xy(dataThis)
    xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
    return ar1D.fit(xTrain, yTrain)


# Compute the relative fitness error of the AR(1) process
def ar1_testerr(data, settings, testFrac=0.1):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 1, 3)
    dataThis = zscore(dataFlat)

    ar1D = AR1D(nHist=1)
    x, y = ar1D.data2xy(dataThis)
    xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
    ar1D.fit(xTrain, yTrain)

    #yHatTrain = ar1D.predict(xTrain)
    #trainErr = ar1D.rel_err(yTrain, yHatTrain)

    yHatTest = ar1D.predict(xTest)
    testErr = ar1D.rel_err(yTest, yHatTest)
    return testErr


# Compute the relative fitness error of the AR(n) process for a few small n values
def ar_testerr(data, settings, testFrac=0.1, maxHist=10):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 1, 3)
    dataThis = zscore(dataFlat)

    testErrLst = []
    for iHist in range(1, maxHist+1):
        ar1D = AR1D(nHist=iHist)
        x, y = ar1D.data2xy(dataThis)
        xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
        ar1D.fit(xTrain, yTrain)

        #yHatTrain = ar1D.predict(xTrain)
        #trainErr = ar1D.rel_err(yTrain, yHatTrain)

        yHatTest = ar1D.predict(xTest)
        testErrLst += [ar1D.rel_err(yTest, yHatTest)]
    return np.array(testErrLst)
