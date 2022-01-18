import os
import shutil
import subprocess
from time import sleep

import numpy as np
from pyne.endf import Evaluation

from src.setting import *
from src.gendf import GENDF
from src.njoy import NjoyInput
from src.cndl import CNDL

getSetting("settings/setting.txt", ENV)
getSettingMul("settings/thermal_setting.txt", THERMAL)

target_list_thermal = os.listdir(ENV["thermal_path"])
target_list_endf = os.listdir(ENV["endf_path"])

for target_thermal in target_list_thermal:
    # read ENDF file
    endf_thermal = Evaluation(os.path.join(ENV["thermal_path"], target_thermal), verbose=False)
    endf_thermal.read()
    if len(endf_thermal.thermal_inelastic) < 1:
        raise TypeError("ENDF file " + target_thermal + " is not the thermal scattering law")
    za_thermal = endf_thermal.target["ZA"]

    # find corresponding material
    za_corr = int(THERMAL[str(za_thermal)][0])
    for target in target_list_endf:
        endf_data = Evaluation(os.path.join(ENV["endf_path"], target), verbose=False)
        #endf_data.read()
        za_target = endf_data.target["ZA"]
        if za_target == za_corr:
            break

    if za_target != za_corr:
        raise OSError("No ENDF file of ZA={} ")
