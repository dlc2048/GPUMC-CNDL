import os
import shutil
import subprocess
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from pyne.endf import Evaluation

from src.setting import *
from src.gendf import GENDF
from src.njoy import NjoyInput
from src.cndl import CNDL
from src.algorithm import interp1d, interp2d
from src.physics import *

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
        za_target = endf_data.target["ZA"]
        if za_target == za_corr:
            endf_data.read()
            break
    
    mat = endf_data.material

    if za_target != za_corr:
        raise OSError("No ENDF file of ZA={} ")

    # get S(a,b) kernel data
    temp = endf_thermal.thermal_inelastic['temperature']
    lnS = endf_thermal.thermal_inelastic['ln(S)']
    alpha = endf_thermal.thermal_inelastic['alpha']
    beta = endf_thermal.thermal_inelastic['beta']
    table = endf_thermal.thermal_inelastic['scattering_law']
    symmetric = endf_thermal.thermal_inelastic['symmetric']
    
    # interpolating S(a,b) kernel to target temperature
    temp_target = float(ENV["temperature"])
    if temp_target < temp[0] - 20 or temp_target > temp[-1] + 20: # tolerance
        raise ValueError("out of range of S(a,b) kernel temperature lists")
    if temp_target < temp[0]:
        table_target = table[:,:,0]
    elif temp_target > temp[-1]:
        table_target = table[:,:,-1]
    else:
        ind = np.argmax(temp_target < temp)
        int_law = int(THERMAL[str(za_thermal)][1])
        ftn = lambda x1, x2, y1, y2, t: interp1d([x1, x2], [y1, y2], int_law).get(t)
        vfunc = np.vectorize(ftn)
        table_target = vfunc(temp[ind-1], temp[ind], table[:,:,ind-1], table[:,:,ind], temp_target)

    # build scattering kernel
    kernel = ScatteringKernel(alpha, beta, table_target, lnS, symmetric)

    # njoy data processing
    print("*** NJOY processing ***")
    os.makedirs(ENV["njoy_workspace"], exist_ok=True)
    shutil.copy(os.path.join(ENV["endf_path"], target),
                os.path.join(ENV["njoy_workspace"], ENV["njoy_target"]))
    ninput = NjoyInput(os.path.join(ENV["njoy_workspace"], ENV["njoy_input"]))
    ninput.setEnv(mat, 293.6)
    ninput.moder(20, -21)
    ninput.reconr(-21, -22, 0.0005)
    ninput.broadr(-21, -22, -23, 0.0005)
    ninput.thermr(0, -23, -24, 1, 0, 0.005, 5)
    ninput.groupr(-21, -24, -30, 10, 6, 3, 8, 1e7)
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

    # convert endf-gendf to cndl structure
    cndl = CNDL(endf_data, gendf_data, verbose=True, MF7=kernel)
    cndl.genEquiProb(verbose=True)
    print("*** WRITE CNDL FILE OF MAT {} ***".format(za_thermal))
    cndl.write(os.path.join("out", "{}.bin".format(za_thermal)), True)
    
