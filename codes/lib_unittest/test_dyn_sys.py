# import standard libraries
import sys
from os.path import dirname, abspath, join

# Export library path
thispath   = dirname(abspath(__file__))
parentpath = dirname(thispath)
libpath    = join(parentpath, 'lib')
sys.path.append(libpath)

# import special libraries
from models.dyn_sys import DynSys

# Set parameters
param = {
    'ALPHA'   : 0.9,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 4000, # Number of timesteps
    'T'       : 100,  # Period of input oscillation
    'STD'     : 0.2,  # STD of neuron noise
    'MAG'     : 1.0   # Magnitude of the periodic input
}

# Create dynamical system
DS1 = DynSys(param)

# Save simulation and metadata
DS1.save("testDynSys.h5")

#Plot results
DS1.plot()
