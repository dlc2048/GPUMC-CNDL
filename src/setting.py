import numpy as np

ENV = {}
ISOTOPES = {}
THERMAL = {}

def getSetting(file_name, setting_dict):
    with open(file_name) as file:
        lines = file.readlines()
        file.close()
    for line in lines:
        item = line.split()
        setting_dict[item[0]] = item[1]

def getSettingMul(file_name, setting_dict):
    with open(file_name) as file:
        lines = file.readlines()
        file.close()
    for line in lines:
        item = line.split()
        setting_dict[item[0]] = item[1:]    
