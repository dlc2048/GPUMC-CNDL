#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Check elastic scattering angular distribution sampling law
"""

import sys
import os
# idle
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# vscode
#par_dir = os.getcwd()
sys.path.append(par_dir)

from pyne.endf import Evaluation

from src.setting import *
from src.physics import *

os.chdir("..")
getSetting("settings/setting.txt", ENV)

mode = "list"

if mode == "list":
    target_list_endf = os.listdir(ENV["endf_path"])

    for target in target_list_endf:
        endf_data = Evaluation(os.path.join(ENV["endf_path"], target), verbose=False)
        endf_data.read()
        print(endf_data)
        print(endf_data.reactions[2].angular_distribution.center_of_mass)
        print(endf_data.reactions[2].angular_distribution.type)
        print("############################################")

elif mode == "target":
    endf_target = Evaluation(os.path.join(ENV["endf_path"], ENV["endf_target"]), verbose=False)
    endf_target.read()
    A = endf_target.target["mass"] / endf_target.projectile["mass"]
    mfad = MF4AngularDistribution(A, endf_target.reactions[2].angular_distribution)
    abin = mfad.getEquiAngularBin(1.3e6, 1.28e6, 1.3e6, 32)

print("finished")
    
