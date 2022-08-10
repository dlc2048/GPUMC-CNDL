#!/usr/bin/env python3
"""Convert every ENDF documents that are written in
settings/thermal_setting.txt to CNDL binary file
target ENDF file should have corresponding MF7 kernel data file
in endf/kernel/ directory

this code is part of the GPU-accelerated Monte Carlo project
"""

import os
import shutil
import subprocess
from time import sleep

import numpy as np
from pyne.endf import Evaluation

from lib.Python.setting import *
from lib.Python.gendf import GENDF
from lib.Python.njoy import NjoyInput
from lib.Python.cndl import CNDL
from lib.Python.physics import *

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


if __name__ == "__main__":
    getSetting("settings/setting.txt", ENV)
    getSettingMul("settings/thermal_setting.txt", THERMAL)

    target_list_thermal = os.listdir(ENV["thermal_path"])
    target_list_endf = os.listdir(ENV["endf_path"])

    egn = np.load(ENV["njoy_group"])

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
            za_target = endf_data.target["ZA"]
            if za_target == za_corr:
                endf_data.read()
                break
        
        mat = endf_data.material

        if za_target != za_corr:
            raise OSError("No ENDF file of ZA={} ")

        # njoy data processing
        print("*** NJOY processing ***")

        thermal_mat = int(THERMAL[str(za_thermal)][1])
        thermal_temp = float(THERMAL[str(za_thermal)][2])

        os.makedirs(ENV["njoy_workspace"], exist_ok=True)
        shutil.copy(os.path.join(ENV["endf_path"], target),
                    os.path.join(ENV["njoy_workspace"], ENV["njoy_target"]))
        shutil.copy(os.path.join(ENV["thermal_path"], target_thermal),
                    os.path.join(ENV["njoy_workspace"], ENV["njoy_kernel"]))
        ninput = NjoyInput(os.path.join(ENV["njoy_workspace"], ENV["njoy_input"]))
        ninput.setEnv(mat, 293.6)
        ninput.setGroup(egn)
        ninput.moder(20, -21)
        ninput.reconr(-21, -22, 0.0005)
        ninput.broadr(-21, -22, -23, 0.0005)
        ninput.thermr(40, -23, -24, thermal_mat, 2, 0, 0.005, 4)
        ninput.groupr(-21, -24, -30, 1, 6, 7, 9, 1e7)
        ninput.moder(-30, 31)
        ninput.stop()
        ninput.write()

        os.chdir(ENV["njoy_workspace"])
        with subprocess.Popen([ENV["njoy_executable"],
                               "-i", ENV["njoy_input"],
                               "-o", ENV["njoy_output"]]) as process:
            while True:
                if process.poll() == 0:
                    break
                elif process.poll() == None:
                    sleep(0.2)
                else:
                    raise OSError

        os.remove(ENV["njoy_output"])
        os.chdir("..")

        print("*** READ GENDF FILE ***")
        gendf_data = GENDF(os.path.join(ENV["njoy_workspace"], ENV["njoy_GENDF"]))
        gendf_data.dropInvalidMF()
        
        # convert endf-gendf to cndl structure
        cndl = CNDL(endf_data, gendf_data, verbose=True, MF7=int(THERMAL[str(za_thermal)][3]))
        cndl.genAliasTable(verbose=True)
        cndl.genEquiProb(verbose=True, alias=True)
        print("*** WRITE CNDL FILE OF MAT {} ***".format(za_thermal))
        cndl.write(os.path.join(ENV["output_path"], "{}.bin".format(za_thermal)), get_reactions_list=True, alias=True)
        
