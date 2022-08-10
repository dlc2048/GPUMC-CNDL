#!/usr/bin/env python3
"""Get the phase-space distribution of 
the secondary proton in the case of neutron-hydrogen recoil,
and the deposition energy in the case of heavier nucleus

this code is part of the GPU-accelerated Monte Carlo project
"""

import os
from time import sleep

import numpy as np

from lib.Python.setting import *
from lib.Python.physics import *
from lib.Python.gpundl import GPUNDL

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


if __name__ == "__main__":
    getSetting("settings/setting.txt", ENV)
    root = ENV["output_path"]
    target_list = os.listdir(root)

    egn = np.load(ENV["njoy_group"])

    for target in target_list[:]:
        # load gpundl file
        print("*** READ GPUNDL FILE ***")
        file = GPUNDL(os.path.join(root, target), verbose=False, alias=True)
        file.getNeutronEnergyGroup("settings/egn.npy")
        file.getPhotonEnergyGroup("settings/egg.npy")
        print("*** COMPUTE ELASTIC SCATTERING SECONDARY ***")
        file.computeElasticSecondary()
        print("*** WRITE CNDL FILE OF MAT {} ***".format(file.za))
        file.write(os.path.join(root, target))

        
