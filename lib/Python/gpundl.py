#!/usr/bin/env python3
"""GPU-side NDL (GPUNDL) binary file interpreter.
privides cross-section and energy-angle distribution plotter
and some sampling algorithm for debugging

this code is part of the GPU-accelerated Monte Carlo project
"""

import numpy as np
from tqdm import tqdm

from lib.Python.setting import *
from lib.Python import gendf
from lib.Python.binary_io import GpumcBinary
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
    def __init__(self, mt, sampling_rule, mul_inv):
        """GPUNDL neutron reaction"""
        self.mt = mt
        self.mf = {}
        self._sampling_rule = sampling_rule
        self._mul_inv = mul_inv

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
        file = GpumcBinary(file_name, mode="r")
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
        file = GpumcBinary(file_name, mode="r")
        self.reactions = {}

        # read atomic mass
        self.za = file.read()[0]
        self.mass = file.read()[0]

        # read total xs
        self.reactions[1] = Reaction(1, None, None)
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
            self.reactions[mt] = Reaction(mt, sampling_rule, mul_inv)
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

    def write(self, file_name):
        file = GpumcBinary(file_name, mode="w")
        # save ZA and atomic mass
        file.write(np.array([self.za], dtype=np.int32))
        file.write(np.array([self.mass], dtype=np.float32))
        # save total xs
        file.write(self.reactions[1].mf[3].xs.astype(np.float32))
        # save MT table & mt alias table
        file.write(self._mt_target)
        file.write(self._mt_alias_table)
        file.write(self._mt_alias_index)

        for i, mt in enumerate(self._mt_target):
            # write sampling law and inverse multiplicity
            file.write(self.reactions[mt]._sampling_rule.astype(np.int32))
            file.write(self.reactions[mt]._mul_inv.astype(np.float32))
        
            # for each reaction, write target tape and probability map
            if 27 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[27].target_tape_alias.astype(np.int32))
                file.write(self.reactions[mt].mf[27].prob_map_alias.astype(np.float32))
                file.write(self.reactions[mt].mf[27].index_map_alias.astype(np.int32))
            if 6 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[6].target_tape_alias.astype(np.int32))
                file.write(self.reactions[mt].mf[6].equiprob_map.astype(np.float32))
                file.write(self.reactions[mt].mf[6].index_map_alias.astype(np.int32))
            if 21 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[21].target_tape_alias.astype(np.int32))
                file.write(self.reactions[mt].mf[21].equiprob_map.astype(np.float32))   
                file.write(self.reactions[mt].mf[21].index_map_alias.astype(np.int32))   
            if 16 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[16].target_tape_alias.astype(np.int32))
                file.write(self.reactions[mt].mf[16].prob_map_alias.astype(np.float32))
                file.write(self.reactions[mt].mf[16].index_map_alias.astype(np.int32))          
        file.close()

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

    def spectrumWeight(self, energy):
        """using 1/E weight function 
        (normalized to 1 when energy is 1 keV)
        """
        return 1e3 / energy
        # return 1

    def computeElasticSecondary(self):
        """compute energy-angle distribution of secondary
        if target is hydrogen, secondary is proton
        otherwise, secondary is energy deposition
        """
        if 2 not in self.reactions:
            raise Exception("Elastic scattering (MT=2) data missing!")
        if self.za in (1001, 101):  # hydrogen recoil
            self._computeHydrogenRecoil()
        else:
            self._computeDoseRecoil()

    def _computeHydrogenRecoil(self):

        # get parameters
        ngroup = len(self.egn) - 1
        thres = float(ENV["recoil_thres"])
        nsample = int(ENV["recoil_nsample"])
        equiprob_nbin = int(ENV["equiprob_nbin"])


        # modify sampling rule
        reaction = self.reactions[2]
        rule = np.copy(reaction._sampling_rule)
        rule = np.append(rule, 2)
        reaction._sampling_rule = rule
        mul_inv = np.copy(reaction._mul_inv)

        # initialize alias map
        target_tape_alias = -np.ones((ngroup, 3), dtype=np.int32)
        target_tape_alias[:,-1] = 0
        prob_map_alias = np.empty((0,equiprob_nbin + 2), dtype=np.float64)
        index_map_alias = np.empty(0, dtype=np.int32)
        trans_matrix = reaction.mf[6].getTransMatrix(equiprob=True)

        # compute secondary energy distribution (alias)
        for i in tqdm(range(ngroup)):
            if self.egn[i] < thres:
                mul_inv[i] = 1.0
                continue
            energy_inc = np.linspace(self.egn[i], self.egn[i+1], nsample+1)
            energy_inc = (energy_inc[1:] + energy_inc[:-1]) / 2
            group_target = np.where(trans_matrix[i,:,0] > 0)[0]

            prob_seg = np.zeros(ngroup + 1, dtype=np.float64)
            
            for gout in group_target:
                energy_out = np.linspace(self.egn[gout], self.egn[gout+1], nsample+1)
                energy_out = (energy_out[1:] + energy_out[:-1]) / 2

                # calculate the group transition probability of recoil particle
                pss = np.zeros(ngroup + 1, dtype=np.float64)
                for ein in energy_inc:
                    ediff = ein - energy_out
                    weight = self.spectrumWeight(ein) * self.spectrumWeight(energy_out)
                    vfunc = np.vectorize(lambda ene: np.argmin(ene > self.egn))
                    ediff_group = vfunc(ediff) - 1
                    ediff_group[ediff < 0] = -1
                    pss[ediff_group] += weight
                
                pss /= np.sum(pss)
                pss *= trans_matrix[i,gout,0]

                prob_seg += pss
                
            # normalize
            prob_seg = prob_seg[:-1]
            prob_seg /= np.sum(prob_seg)

            # set alias table
            sec_group_target = np.where(prob_seg > 0)[0]
            group_floor = sec_group_target[0]
            prob_seg = prob_seg[sec_group_target[0]:sec_group_target[-1]+1]
            domain = np.arange(0, prob_seg.shape[0], 1)
            alias = AliasTable(domain, prob_seg)
            # proton and neutron are have same angular distribution
            adist = np.copy(trans_matrix[i,sec_group_target[0]:sec_group_target[-1]+1])
            adist[:,0] = alias.getProbTable()

            # set tapes
            target_tape_alias[i] = [len(prob_map_alias), group_floor, len(prob_seg)]
            prob_map_alias = np.append(prob_map_alias, adist, axis=0)
            index_map_alias = np.append(index_map_alias, alias.getAliasTable())
        
        reaction._mul_inv = mul_inv
        self.reactions[2].mf[21] = MF6Like(target_tape_alias, prob_map_alias, 21, index_map_alias)

    def _computeDoseRecoil(self):
        
        # get parameters
        ngroup = len(self.egn) - 1
        thres = float(ENV["recoil_thres"])
        nsample = int(ENV["recoil_nsample"])

        # modify sampling rule
        reaction = self.reactions[2]
        rule = np.copy(reaction._sampling_rule)
        rule[0] = 1
        reaction._sampling_rule = rule

        # initialize alias map
        target_tape_alias = -np.ones((ngroup, 3), dtype=np.int32)
        target_tape_alias[:,-1] = 0
        prob_map_alias = np.empty(0, dtype=np.float64)
        index_map_alias = np.empty(0, dtype=np.int32)
        trans_matrix = reaction.mf[6].getTransMatrix()[:,:,0]

        # compute secondary energy distribution (alias)
        for i in tqdm(range(ngroup)):
            if self.egn[i] < thres:
                continue
            energy_inc = np.linspace(self.egn[i], self.egn[i+1], nsample+1)
            energy_inc = (energy_inc[1:] + energy_inc[:-1]) / 2
            group_target = np.where(trans_matrix[i] > 0)[0]

            prob_seg = np.zeros(ngroup + 1, dtype=np.float64)
            
            for gout in group_target:
                energy_out = np.linspace(self.egn[gout], self.egn[gout+1], nsample+1)
                energy_out = (energy_out[1:] + energy_out[:-1]) / 2

                # calculate the group transition probability of recoil particle
                pss = np.zeros(ngroup + 1, dtype=np.float64)
                for ein in energy_inc:
                    ediff = ein - energy_out
                    weight = self.spectrumWeight(ein) * self.spectrumWeight(energy_out)
                    vfunc = np.vectorize(lambda ene: np.argmin(ene > self.egn))
                    ediff_group = vfunc(ediff) - 1
                    ediff_group[ediff < 0] = -1
                    pss[ediff_group] += weight
                
                pss /= np.sum(pss)
                pss *= trans_matrix[i,gout]

                prob_seg += pss
                
            # normalize
            prob_seg = prob_seg[:-1]
            prob_seg /= np.sum(prob_seg)

            # set alias table
            sec_group_target = np.where(prob_seg > 0)[0]
            group_floor = sec_group_target[0]
            prob_seg = prob_seg[sec_group_target[0]:sec_group_target[-1]+1]
            domain = np.arange(0, prob_seg.shape[0], 1)
            alias = AliasTable(domain, prob_seg)

            # set tapes
            target_tape_alias[i] = [len(prob_map_alias), group_floor, len(prob_seg)]
            prob_map_alias = np.append(prob_map_alias, alias.getProbTable())
            index_map_alias = np.append(index_map_alias, alias.getAliasTable())

        reaction.mf[27] = MF6Like(target_tape_alias, prob_map_alias, 27, index_map_alias)
