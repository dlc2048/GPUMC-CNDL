#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Binary decode test for GPUMC CNDL
"""

import sys
import os

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(par_dir)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.gpundl import GPUNDL
from src.setting import *

os.chdir("..")
getSetting("settings/setting.txt", ENV)

# import material 4009 binary (Beryllium 9)
mat = GPUNDL("out/4009.bin", alias=True)
mat.getNeutronEnergyGroup("settings/egn.npy")
mat.getPhotonEnergyGroup("settings/egg.npy")

n = 200000

dump = []
for i in tqdm(range(n)):
    dump += mat2.sampling(200)
dump = np.array(dump)

mf = 6
energy, counts = np.unique(dump[dump[:,0]==mf,1], return_counts=True)

hist = np.zeros(len(mat.egn)-1)
hist[energy.astype(np.int32)] = counts

plt.step(mat.egn[:-1], hist / (mat.egn[1:] - mat.egn[:-1]))
plt.xscale("log")
plt.show()

plt.hist(dump[dump[:,0]==mf,2], bins=41)
plt.show()

#np.save("dump", dump)

