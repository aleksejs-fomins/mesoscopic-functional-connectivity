import numpy as np
import matplotlib.pyplot as plt

from codes.lib.plots.matplotblib_lib import hist_int

##########################
# Integer Histogram
#########################

minlag = 1
maxlag = 5
mShape = (12, 12)

# Construct fake data matrix and p-value matrix
M = np.random.randint(minlag, maxlag+1, mShape).astype(float)
P = np.random.uniform(0, 1, mShape)

# Set all unlikely values of M to NAN
M[P < 0.1] = np.nan

# Select all off-diagonal values of M that are not NAN
offDiagIdx = ~np.eye(M.shape[0], dtype=bool)
notNanIdx = ~np.isnan(M)
M1dNotNan = M[offDiagIdx & notNanIdx]

# Make histogram
fig, ax = plt.subplots()
hist_int(ax, M1dNotNan, minlag, maxlag)
plt.show()
