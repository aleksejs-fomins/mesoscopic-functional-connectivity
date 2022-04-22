# Append base directory
import os,sys,inspect
rootname = "mesoscopic-functional-connectivity"
thispath = os.getcwd()
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
mc = MetricCalculator(serial=True, verbose=False)

import code_python.lib.null.axonal_simulations as axsim


axsim.crosstalk_test_two_node_negative(mc, ['crosscorr', 'MultivariateTE'], testConv=True, nCTDiscr=10,
                  nTest=50, maxLag=5, pTHR=0.01, noiseFrac=0.01, convDT=0.05, convTau=0.5, showPlot=False)
