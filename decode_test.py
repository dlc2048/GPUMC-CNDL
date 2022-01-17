import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.gpundl import GPUNDL
from src.setting import *

getSetting("settings/setting.txt", ENV)

mat = GPUNDL("out/13027.bin")
mat.getNeutronEnergyGroup("settings/egn.npy")
mat.getPhotonEnergyGroup("settings/egg.npy")

#mat.plotEnergyDist(103, 27, 100)
"""
dump = []
n = 1000000
for i in tqdm(range(n)):
    dump += mat.sampling(171)
    
dump = np.array(dump)

mf = 6
energy, counts = np.unique(dump[dump[:,0]==mf,1], return_counts=True)

hist = np.zeros(len(mat.egn)-1)
hist[energy.astype(np.int32)] = counts

plt.step(mat.egn[:-1], hist / (mat.egn[1:] - mat.egn[:-1]))
plt.xscale("log")
plt.show()

plt.hist(dump[dump[:,0]==mf,2], bins=100)
plt.show()

np.save("dump", dump)
"""
