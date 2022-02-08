import os
import shutil
import subprocess
from time import sleep

import numpy as np
from pyne.endf import Evaluation
from scipy.special import eval_legendre
import matplotlib.pyplot as plt

from src.setting import *
from src.gendf import GENDF
from src.njoy import NjoyInput
from src.cndl import CNDL

getSetting("settings/setting.txt", ENV)

target_list = os.listdir(ENV["endf_path"])

# read ENDF file

endf_data = Evaluation(os.path.join(ENV["endf_path"], ENV["endf_target"]), verbose=False)
endf_data.read()
mat = endf_data.material
temp = float(ENV["temperature"])
    
# njoy data processing

print("*** NJOY processing ***")
os.makedirs(ENV["njoy_workspace"], exist_ok=True)
shutil.copy(os.path.join(ENV["endf_path"], ENV["endf_target"]),
            os.path.join(ENV["njoy_workspace"], ENV["njoy_target"]))
ninput = NjoyInput(os.path.join(ENV["njoy_workspace"], ENV["njoy_input"]))
ninput.setEnv(mat, temp)
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

# read GENDF file
print("*** READ GENDF FILE ***")
gendf_data = GENDF(os.path.join(ENV["njoy_workspace"], ENV["njoy_GENDF"]))
# convert endf-gendf to cndl structure

cndl = CNDL(endf_data, gendf_data, verbose=True)

cndl.genEquiProb(verbose=True)
print("*** WRITE CNDL FILE OF MAT {} ***".format(cndl.za))
cndl.write(os.path.join("out", "{}.bin".format(cndl.za)), True)
