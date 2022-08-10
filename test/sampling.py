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

from lib.Python.gpundl import GPUNDL
from lib.Python.setting import *

os.chdir("..")
getSetting("settings/setting.txt", ENV)

# import material 4009 binary (Beryllium 9)
mat = GPUNDL("out/8016.bin", alias=True)
mat.getNeutronEnergyGroup("settings/egn.npy")
mat.getPhotonEnergyGroup("settings/egg.npy")

n = 200000

dump = []
for i in tqdm(range(n)):
    dump += mat.reactions[2].sampling(158)
dump = np.array(dump)

mf = 6
energy, counts = np.unique(dump[dump[:,0]==mf,1], return_counts=True)

ebin = (mat.egn[1:] + mat.egn[:-1]) / 2
ee = ebin[energy.astype(np.int32)] / 1e6

print(np.sum(ee * counts) / np.sum(counts))

mf = 27
energy, counts = np.unique(dump[dump[:,0]==mf,1], return_counts=True)

ebin = (mat.egn[1:] + mat.egn[:-1]) / 2
ee = ebin[energy.astype(np.int32)] / 1e6

print(np.sum(ee * counts) / np.sum(counts))
"""
hist = np.zeros(len(mat.egn)-1)
hist[energy.astype(np.int32)] = counts

plt.step(mat.egn[:-1], hist / (mat.egn[1:] - mat.egn[:-1]))
plt.xscale("log")
plt.show()

plt.hist(dump[dump[:,0]==mf,2], bins=41)
plt.show()

#np.save("dump", dump)
"""
