# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/'), 'sim_ds_h5')

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
from models.dyn_sys import DynSys


# Set parameters
param = {
    'ALPHA'   : 0.1,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 4000, # Number of timesteps
    'MAG'     : 0,    # Magnitude of input
    'T'       : 20,   # Period of input oscillation
    'STD'     : 0.2   # STD of neuron noise
}

# Create dynamical system
DS1 = DynSys(param)

# Save simulation and metadata
DS1.save(os.path.join(rezPath, "sim_dynsys_1.h5"))

#Plot results
DS1.plot()
