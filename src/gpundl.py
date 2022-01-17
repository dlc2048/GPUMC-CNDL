import numpy as np

from src.setting import *
from src import gendf
from src.binary_io import NdlBinary


class MF6Like(gendf.MF6Like):
    def __init__(self, target_tape, equiprob_map, mf):
        self._mf = mf
        self.target_tape = target_tape
        self.equiprob_map = equiprob_map
        if mf == 27:
            self.prob_map = equiprob_map
        else:
            self.prob_map = equiprob_map[:,:1]

    def setFromMatrix(self, matrix): # override method
        raise AttributeError("'MF6Like' object has no attribute 'setFromMatrix'")

    def genEquiProbMap(self): #override method
        raise AttributeError("'MF6Like' object has no attribute 'genEquiProbMap'")

    def sampling(self, inc_group):
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

            if group_pointer > 187:
                raise ValueError

            return [[self._mf, group_pointer, mu]]


class MF16(gendf.MF16):
    def __init__(self, target_tape, prob_map):
        self._mf = 16
        self.target_tape = target_tape
        self.prob_map = prob_map 

    def sampling(self, inc_group):
        # sampling the number of photon
        line_start, multiplicity, group_start = self.target_tape[inc_group]
        if line_start < 0:
            return []
        multiplicity = np.frombuffer(self.target_tape[inc_group,1:2].tobytes(),
                                     dtype=np.float32)[0]
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
                if rand < self.prob_map[line_pointer]:
                    break
                line_pointer += 1
                group_pointer += 1

            # sampling directional cosine
            mu = np.random.random() * 2 - 1
            photon += [[self._mf, group_pointer, mu]]
            multiplicity -= 1
        return photon

class Reaction(gendf.Reaction):
    def __init__(self, mt, sampling_rule):
        self.mt = mt
        self.mf = {}
        self._sampling_rule = sampling_rule

    def _add(self, egn, mf, nz, lrflag, data, label):
        raise AttributeError("'Reaction' object has no attribute '_add'")

    def sampling(self, inc_group):
        exit_particles = []
        for i in range(0, len(self._sampling_rule), 2):
            if self._sampling_rule[i+1] < 0:
                continue
            particle_temp = self.mf[self._sampling_rule[i]].sampling(inc_group)
            if len(particle_temp) > 0:
                exit_particles += particle_temp
        return exit_particles


class GPUNDL(gendf.GendfInterface):
    def __init__(self, file_name, verbose=False):
        super().__init__()

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

    def getNeutronEnergyGroup(self, file_name):
        self.egn = np.load(file_name)

    def getPhotonEnergyGroup(self, file_name):
        self.egg = np.load(file_name)

    def sampling(self, inc_group):
        # get type of reaction
        rand = np.random.random()
        pointer = 0
        while True:
            if rand < self._mt_map[inc_group,pointer]:
                break
            pointer += 1
        mt = self._mt_target[pointer]
        return self.reactions[mt].sampling(inc_group)