import os

import numpy as np
import matplotlib.pyplot as plt
from pyne.endf import Evaluation

from src.setting import *
from src.physics import MF4AngularDistribution
from src.algorithm import interp1d

getSetting("settings/setting.txt", ENV)


target_list_endf = os.listdir(ENV["endf_path"])
"""
for target in target_list_endf:
    endf_data = Evaluation(os.path.join(ENV["endf_path"], target), verbose=False)
    endf_data.read()
    print(endf_data)
    print(endf_data.reactions[2].angular_distribution.center_of_mass)
    print(endf_data.reactions[2].angular_distribution.type)
    print("############################################")
"""

endf_target = Evaluation("endf/n/n_0425_4-Be-9.dat", verbose=False)
endf_target.read()
mfad = MF4AngularDistribution(endf_target.target['mass'],
                              endf_target.reactions[2].angular_distribution)

x = mfad._ad.probability[-1].x
y = mfad._ad.probability[-1].y

area = 0
for i in range(len(x) - 1):
    area += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
