#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Binary decode test for GPUMC CNDL
"""

import sys
import os

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(par_dir)

from src.gpundl import GPUNDL
from src.setting import *

os.chdir("..")
getSetting("settings/setting.txt", ENV)

# import material 4009 binary (Beryllium 9)
mat = GPUNDL("out/101.bin")
mat.getNeutronEnergyGroup("settings/egn.npy")
mat.getPhotonEnergyGroup("settings/egg.npy")

# print reactions list
print(mat.reactions)
"""
# plot neutron energy distribution of (elastic) and (n,2n) reactions
# when the group of incident neutron beam is 171 (mean=7.32 MeV)
mat.plotEnergyDist(2, 6, 171)
mat.plotEnergyDist(16, 6, 171)
mat.show()

# plot neutron angular distribution of (elastic) reaction
# when the group of incident neutron beam is 171 (mean=7.32 MeV)
mat.plotAngularDist(2, 6, 171, 171, True)
mat.plotAngularDist(2, 6, 171, 170, True)
mat.plotAngularDist(2, 6, 171, 169, True)
mat.plotAngularDist(2, 6, 171, 168, True)
mat.plotAngularDist(2, 6, 171, 167, True)
mat.show()

# plot gamma multiplicity of (z,absorp) and (n,gamma) reactions
mat.plotGammaMultiplicity(27)
mat.plotGammaMultiplicity(102)
mat.show()

# plot gamma spectrum of (n,gamma) reaction
# when the group of incident neutron beam is 150
mat.plotGammaSpectrum(102, 150)
mat.show()
"""
