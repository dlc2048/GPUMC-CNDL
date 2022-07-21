#!/usr/bin/env python3
"""GPU-side NDL (GPUNDL) binary file interpreter.
privides cross-section and energy-angle distribution plotter
and some sampling algorithm for debugging

this code is part of the GPU-accelerated Monte Carlo project
"""

import numpy as np

from lib.Python.setting import *
from lib.Python import gendf
from lib.Python.binary_io import NdlBinary
from lib.Python.algorithm import *

__copyright__ = "Copyright 2021, GPUMC Project"
__license__ = "MIT"
__author__ = "Chang-Min Lee"
__email__ = "dlc2048@postech.ac.kr"
__status__ = "Production"


class MF6Like(gendf.MF6Like):
    def __init__(self, target_tape, equiprob_map, mf, index_map=None):
        """GPUNDL MF6 and MF21 (energy-angle distribution)"""
        # initialize
        self.target_tape = None
        self.prob_map = None
        self.target_tape_alias = None
        self.prob_map_alias = None
        self.index_map_alias = None

        self._mf = mf

        if index_map is None:  # cumul mode
            self.target_tape = target_tape
            self.equiprob_map = equiprob_map
            if mf == 27:
                self.prob_map = equiprob_map
            else:
                self.prob_map = equiprob_map[:,:1]

        else:
            self.target_tape_alias = target_tape
            self.equiprob_map = equiprob_map
            if mf == 27:
                self.prob_map_alias = equiprob_map
            else:
                self.prob_map_alias = equiprob_map[:,:1]
            # set alias mode if index_map is not None
            self.index_map_alias = index_map

    def setFromMatrix(self, matrix): # override method
        raise AttributeError("'MF6Like' object has no attribute 'setFromMatrix'")

    def genEquiProbMap(self): #override method
        raise AttributeError("'MF6Like' object has no attribute 'genEquiProbMap'")

    def sampling(self, inc_group):
        """sampling secondary particle"""
        if self.index_map_alias is None:
            return self._samplingFromCumul(inc_group)
        else:
            return self._samplingFromAlias(inc_group)
    
    def _samplingFromCumul(self, inc_group):
        if self._mf == 27:
            # sampling exit channel group
            line_pointer, group_pointer = self.target_tape[inc_group]
            if line_pointer < 0:
                return []
            rand = np.random.random()
            while True:
                if rand < self.equiprob_map[line_pointer]:
                    break
                line_pointer += 1
                group_pointer += 1

            return [[self._mf, group_pointer, -2]]

        else:
            # sampling exit channel group
            line_pointer, group_pointer = self.target_tape[inc_group]
            if line_pointer < 0:
                return []
            rand = np.random.random()
            while True:
                if rand < self.equiprob_map[line_pointer,0]:
                    break
                line_pointer += 1
                group_pointer += 1

            # sampling directional cosine
            rand = np.random.randint(0,int(ENV["equiprob_nbin"]))
            rand2 = np.random.random()
            mu = self.equiprob_map[line_pointer,rand+1] * rand2 \
                +self.equiprob_map[line_pointer,rand+2] * (1-rand2)

            return [[self._mf, group_pointer, mu]]
    
    def _samplingFromAlias(self, inc_group):
        if self._mf == 27:
            # sampling exit channel group
            target_from, group_base, target_len = self.target_tape_alias[inc_group]
            if target_from < 0:
                return []
            rand = np.random.random() * target_len
            group_up = int(rand)
            rand -= group_up
            if rand > self.equiprob_map[target_from + group_up]:
                group_up = self.index_map_alias[target_from + group_up]
            
            return[[self._mf, group_base + group_up, -2]]
        
        else:
            # sampling exit channel group
            target_from, group_base, target_len = self.target_tape_alias[inc_group]
            if target_from < 0:
                return []
            rand = np.random.random() * target_len
            group_up = int(rand)
            rand -= group_up
            if rand > self.equiprob_map[target_from + group_up,0]:
                group_up = self.index_map_alias[target_from + group_up]            

            # sampling directional cosine
            rand = np.random.random() * int(ENV["equiprob_nbin"])
            ind_bin = int(rand)
            rand -= ind_bin
            mu = self.equiprob_map[target_from + group_up,ind_bin+1] * rand \
                +self.equiprob_map[target_from + group_up,ind_bin+2] * (1-rand)
            
            return [[self._mf, group_base + group_up, mu]]

class MF16(gendf.MF16):
    def __init__(self, multiplicity, target_tape, prob_map, index_map=None):
        """GPUNDL MF16 (photon spectrum and multiplicity)"""
        # initialize
        self.target_tape = None
        self.prob_map = None
        self.target_tape_alias = None
        self.prob_map_alias = None
        self.index_map_alias = None

        self._mf = 16

        self.multiplicity = multiplicity

        if index_map is None:
            self.target_tape = target_tape
            self.prob_map = prob_map 
        else:
            self.target_tape_alias = target_tape
            self.prob_map_alias = prob_map
            # set alias mode if index_map is not None
            self.index_map_alias = index_map

    def sampling(self, inc_group):
        """sampling secondary particle"""
        if self.index_map_alias is None:
            return self._samplingFromCumul(inc_group)
        else:
            return self._samplingFromAlias(inc_group)

    def _samplingFromCumul(self, inc_group):
        # sampling the number of photon
        line_start, group_start = self.target_tape[inc_group]
        if line_start < 0:
            return []
        multiplicity = self.multiplicity[inc_group]
        photon = []
        while True:
            rand = np.random.random()
            if rand > multiplicity:
                break
            line_pointer = line_start
            group_pointer = group_start
            # sampling exit channel group
            rand = np.random.random()
            while True:
                if rand < self.prob_map[line_pointer,0]:
                    break
                line_pointer += 1
                group_pointer += 1

            # sampling directional cosine
            mu = np.random.random() * 2 - 1
            photon += [[self._mf, group_pointer, mu]]
            multiplicity -= 1
        return photon
    
    def _samplingFromAlias(self, inc_group):
        # sampling the number of photon
        target_from, group_base, target_len = self.target_tape_alias[inc_group]
        if target_from < 0:
            return []
        multiplicity = self.multiplicity[inc_group]
        photon = []
        while True:
            rand = np.random.random()
            if rand > multiplicity:
                break
            rand = np.random.random() * target_len
            group_up = int(rand)
            rand -= group_up
            if rand > self.prob_map_alias[target_from + group_up,0]:
                group_up = self.index_map_alias[target_from + group_up]
            
            # sampling directional cosine
            mu = np.random.random() * 2 - 1
            photon += [[self._mf, target_from + group_up, mu]]
            multiplicity -= 1
        return photon


class Reaction(gendf.Reaction):
    _QID_MAP = {1: 6, 2:21, 3:16}
    def __init__(self, mt, sampling_rule):
        """GPUNDL neutron reaction"""
        self.mt = mt
        self.mf = {}
        self._sampling_rule = sampling_rule

    def _add(self, egn, mf, nz, lrflag, data, label):  # override
        raise AttributeError("'Reaction' object has no attribute '_add'")

    def sampling(self, inc_group):
        """sampling secondaries by using sampling rule of
        target neutron interaction
        """
        exit_particles = []
        if self._sampling_rule[0]:
            particle_temp = self.mf[27].sampling(inc_group)
            exit_particles += particle_temp

        for i in self._sampling_rule[1:]:
            particle_temp = self.mf[self._QID_MAP[i]].sampling(inc_group)
            if len(particle_temp) > 0:
                exit_particles += particle_temp
        return exit_particles


class GPUNDL(gendf.GendfInterface):
    def __init__(self, file_name, verbose=False, alias=False):
        """GPUNDL binary file interpreter

        alias: using alias table sampling scheme if true
        using cumulative sampling scheme elsewhere
        """
        super().__init__(alias)

        self._is_alias = alias

        if alias:
            self._readAlias(file_name)
        else:
            self._readCumul(file_name)

    def _readCumul(self, file_name):
        file = NdlBinary(file_name, mode="r")
        self.reactions = {}

        # read atomic mass
        self.za = file.read()[0]
        self.mass = file.read()[0]

        # read total xs
        self.reactions[1] = Reaction(1, None)
        self.reactions[1].mf[3] = gendf.MF3(file.read())

        # read MT table & mt sampling table
        self._mt_target = file.read()
        self._mt_map = file.read()

        xs_prob_cumul = np.zeros_like(self.reactions[1].mf[3].xs)
        # read MT data
        for mt in self._mt_target:
            sampling_rule = file.read()
            self.reactions[mt] = Reaction(mt, sampling_rule)
            xs_prob = self._mt_map[:,np.argmax(self._mt_target == mt)] - xs_prob_cumul
            self.reactions[mt].mf[3] = gendf.MF3(xs_prob * self.reactions[1].mf[3].xs)
            xs_prob_cumul = self._mt_map[:,np.argmax(self._mt_target == mt)]
            tindex = 0
            for i in range(0, len(sampling_rule), 2):
                if sampling_rule[i+1] < tindex:
                    continue
                mf = sampling_rule[i]
                target_tape = file.read()
                prob_map = file.read()
                if mf == 16: # photon production
                    self.reactions[mt].mf[mf] = MF16(target_tape, prob_map)
                else:
                    self.reactions[mt].mf[mf] = MF6Like(target_tape, prob_map, mf)
                tindex += 1

    def _readAlias(self, file_name):
        file = NdlBinary(file_name, mode="r")
        self.reactions = {}

        # read atomic mass
        self.za = file.read()[0]
        self.mass = file.read()[0]

        # read total xs
        self.reactions[1] = Reaction(1, None)
        self.reactions[1].mf[3] = gendf.MF3(file.read())

        # read MT table & mt alias table
        self._mt_target = file.read()
        self._mt_alias_table = file.read()
        self._mt_alias_index = file.read()
        
        pyfunc = lambda table, index: probFromAlias(table, index)
        vfunc = np.vectorize(pyfunc, signature='(m),(m)->(n)')
        xs_prob = vfunc(self._mt_alias_table, self._mt_alias_index)

        for i, mt in enumerate(self._mt_target):
            sampling_rule = file.read()
            mul_inv = file.read()
            self.reactions[mt] = Reaction(mt, sampling_rule)
            self.reactions[mt].mf[3] = gendf.MF3(xs_prob[:,i] * self.reactions[1].mf[3].xs)
            if sampling_rule[0]:  # residual dose exist
                target_tape_alias = file.read()
                prob_map_alias = file.read()
                index_map_alias = file.read()
                self.reactions[mt].mf[27] = MF6Like(target_tape_alias, prob_map_alias, 27, index_map_alias)
            for i in (1,2,3):  # secondary neutron, proton and photon
                if i in sampling_rule[1:]:
                    target_tape_alias = file.read()
                    prob_map_alias = file.read()
                    index_map_alias = file.read()
                    if i == 3:  # photon
                        # reconstruct multiplicity
                        mul_max = np.sum(sampling_rule[1:] == 3)
                        multiplicity = mul_max - mul_inv
                        self.reactions[mt].mf[16] = MF16(multiplicity, target_tape_alias, prob_map_alias, index_map_alias)
                    else:
                        mf = 6 if i == 1 else 21
                        self.reactions[mt].mf[mf] = MF6Like(target_tape_alias, prob_map_alias, mf, index_map_alias)

    def getNeutronEnergyGroup(self, file_name, MeV=False):
        self.egn = np.load(file_name)
        if MeV:
            self.egn *= 1e-6

    def getPhotonEnergyGroup(self, file_name, MeV=False):
        self.egg = np.load(file_name)
        if MeV:
            self.egg *= 1e-6

    def sampling(self, inc_group):
        """sampling reaction type and all secondaries"""
        if self._is_alias:
            return self._samplingFromAlias(inc_group)
        else:
            return self._samplingFromCumul(inc_group)

    def _samplingFromAlias(self, inc_group):
        # get type of reaction
        rand = np.random.random() * len(self._mt_target)
        pointer = int(rand)
        rand = rand - pointer
        if rand > self._mt_alias_table[inc_group, pointer]:
            pointer = self._mt_alias_index[inc_group, pointer]
        mt = self._mt_target[pointer]
        return self.reactions[mt].sampling(inc_group)

    def _samplingFromCumul(self, inc_group):
        # get type of reaction
        rand = np.random.random()
        pointer = 0
        while True:
            if rand < self._mt_map[inc_group,pointer]:
                break
            pointer += 1
        mt = self._mt_target[pointer]
        return self.reactions[mt].sampling(inc_group)
