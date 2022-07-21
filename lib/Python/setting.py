#!/usr/bin/env python3
"""GPUMC-CNDL global settings

this code is part of the GPU-accelerated Monte Carlo project
"""

import numpy as np

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


ENV = {}
ISOTOPES = {}
THERMAL = {}

def getSetting(file_name, setting_dict):
    with open(file_name) as file:
        lines = file.readlines()
        file.close()
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == "*": # comment line
            continue
        item = line.split()
        if len(item) == 0:
            continue
        setting_dict[item[0]] = item[1]

def getSettingMul(file_name, setting_dict):
    with open(file_name) as file:
        lines = file.readlines()
        file.close()
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == "*": # comment line
            continue
        item = line.split()
        if len(item) == 0:
            continue
        setting_dict[item[0]] = item[1:]    
