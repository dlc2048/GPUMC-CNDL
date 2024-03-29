#!/usr/bin/env python3
"""Convert every ENDF documents that are written in
settings/stable_isotopes.txt to CNDL binary file

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

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


if __name__ == "__main__":
    getSetting("settings/setting.txt", ENV)
    getSetting("settings/stable_isotopes.dat", ISOTOPES)

    target_list = os.listdir(ENV["endf_path"])

    egn = np.load(ENV["njoy_group"])

    for target in target_list:
        # read ENDF file
        endf_data = Evaluation(os.path.join(ENV["endf_path"], target), verbose=False)
        endf_data.read()
        mat = endf_data.material
        za = endf_data.target["ZA"]
        print("*** MAT {} IS DETECTED ***".format(za))
        if str(endf_data.target["ZA"]) not in ISOTOPES:
            print("*** MAT {} IS UNSTABLE ISOTOPE: SKIPPED ***".format(za))
            continue
        
        # njoy data processing
        print("*** NJOY processing ***")
        os.makedirs(ENV["njoy_workspace"], exist_ok=True)
        shutil.copy(os.path.join(ENV["endf_path"], target),
                    os.path.join(ENV["njoy_workspace"], ENV["njoy_target"]))
        ninput = NjoyInput(os.path.join(ENV["njoy_workspace"], ENV["njoy_input"]))
        ninput.setEnv(mat, 293.6)
        ninput.setGroup(egn)
        ninput.moder(20, -21)
        ninput.reconr(-21, -22, 0.0005)
        ninput.broadr(-21, -22, -23, 0.0005)
        ninput.thermr(0, -23, -24, 0, 1, 0, 0.005, 4)
        ninput.groupr(-21, -24, -30, 1, 6, 7, 8, 1e7)
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

        # read GENDF file
        print("*** READ GENDF FILE ***")
        gendf_data = GENDF(os.path.join(ENV["njoy_workspace"], ENV["njoy_GENDF"]))
        gendf_data.dropInvalidMF()
        
        # convert endf-gendf to cndl structure
        cndl = CNDL(endf_data, gendf_data, verbose=True)
        cndl.genAliasTable(verbose=True)
        cndl.genEquiProb(verbose=True, alias=True)
        print("*** WRITE CNDL FILE OF MAT {} ***".format(za))
        cndl.write(os.path.join(ENV["output_path"], "{}.bin".format(cndl.za)), get_reactions_list=True, alias=True)
